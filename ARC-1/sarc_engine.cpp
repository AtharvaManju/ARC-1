#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <cuda_runtime.h>
#include <cerrno>
#include <cstring>
#include <string>
#include <stdexcept>
#include <atomic>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#if AIMEMORY_WITH_URING
#include <liburing.h>
#endif

#if AIMEMORY_WITH_GDS
#include "cufile.h"
#endif

static constexpr size_t kAlign = 4096;

static inline size_t round_up(size_t x, size_t a = kAlign) { return ((x + a - 1) / a) * a; }
static inline bool is_aligned_u64(uint64_t x, uint64_t a = kAlign) { return (x % a) == 0; }

static void throw_if(bool cond, const std::string& msg) { if (cond) throw std::runtime_error(msg); }
static void throw_if_cuda(cudaError_t st, const char* msg) {
    if (st != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(st));
}

static void pread_full(int fd, void* buf, size_t bytes, off_t off) {
    char* p = (char*)buf;
    size_t done = 0;
    while (done < bytes) {
        ssize_t r = ::pread(fd, p + done, bytes - done, off + (off_t)done);
        if (r == 0) throw std::runtime_error("pread EOF");
        if (r < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("pread failed: ") + strerror(errno));
        }
        done += (size_t)r;
    }
}

static void pwrite_full(int fd, const void* buf, size_t bytes, off_t off) {
    const char* p = (const char*)buf;
    size_t done = 0;
    while (done < bytes) {
        ssize_t w = ::pwrite(fd, p + done, bytes - done, off + (off_t)done);
        if (w == 0) throw std::runtime_error("pwrite 0");
        if (w < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("pwrite failed: ") + strerror(errno));
        }
        done += (size_t)w;
    }
}

static void ensure_file_size_delta_alloc(int fd, off_t end_off) {
    struct stat st;
    if (::fstat(fd, &st) != 0) return;
    if (st.st_size >= end_off) return;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    int rc = ::posix_fallocate(fd, st.st_size, end_off - st.st_size);
    (void)rc;
#endif
    ::ftruncate(fd, end_off);
}

struct PinnedBlock { void* raw=nullptr; void* aligned=nullptr; size_t bytes=0; };

static PinnedBlock alloc_pinned_aligned(size_t bytes) {
    size_t padded = round_up(bytes, kAlign);
    size_t alloc_bytes = padded + kAlign;
    void* raw = nullptr;
    throw_if_cuda(cudaHostAlloc(&raw, alloc_bytes, cudaHostAllocDefault), "cudaHostAlloc");
    uintptr_t base = (uintptr_t)raw;
    uintptr_t aligned = (base + (kAlign - 1)) & ~(uintptr_t)(kAlign - 1);
    return PinnedBlock{raw, (void*)aligned, padded};
}

static void free_pinned(PinnedBlock& b) {
    if (b.raw) cudaFreeHost(b.raw);
    b = PinnedBlock{};
}

class SARCCore {
private:
    bool use_gds_ = false;
    bool use_uring_ = false;

#if AIMEMORY_WITH_URING
    io_uring ring_{};
#endif

    void* pinned_raw_ = nullptr;
    void* staging_aligned_ = nullptr;
    size_t staging_bytes_ = 0;

    std::atomic<bool> last_used_direct_{false};

public:
    SARCCore(size_t staging_mb = 512) {
#if AIMEMORY_WITH_GDS
        CUfileError_t st = cuFileDriverOpen();
        use_gds_ = (st.err == CU_FILE_SUCCESS);
#else
        use_gds_ = false;
#endif

#if AIMEMORY_WITH_URING
        if (io_uring_queue_init(256, &ring_, 0) == 0) use_uring_ = true;
#else
        use_uring_ = false;
#endif

        staging_bytes_ = staging_mb * 1024ull * 1024ull;
        size_t alloc_bytes = staging_bytes_ + kAlign;
        throw_if_cuda(cudaHostAlloc(&pinned_raw_, alloc_bytes, cudaHostAllocDefault), "cudaHostAlloc staging");
        uintptr_t base = (uintptr_t)pinned_raw_;
        uintptr_t aligned = (base + (kAlign - 1)) & ~(uintptr_t)(kAlign - 1);
        staging_aligned_ = (void*)aligned;
    }

    ~SARCCore() {
#if AIMEMORY_WITH_GDS
        if (use_gds_) cuFileDriverClose();
#endif
#if AIMEMORY_WITH_URING
        if (use_uring_) io_uring_queue_exit(&ring_);
#endif
        if (pinned_raw_) cudaFreeHost(pinned_raw_);
    }

    bool gds_enabled() const { return use_gds_; }
    bool uring_enabled() const { return use_uring_; }
    bool last_used_direct() const { return last_used_direct_.load(); }
    size_t staging_capacity_bytes() const { return staging_bytes_; }
    uintptr_t staging_ptr() const { return (uintptr_t)staging_aligned_; }
    size_t round_up_4k(size_t x) const { return round_up(x, kAlign); }

    torch::Tensor alloc_pinned_u8(size_t bytes) {
        size_t padded = round_up(bytes, kAlign);
        auto* blk = new PinnedBlock(alloc_pinned_aligned(padded));
        auto deleter = [blk](void* /*ptr*/) { free_pinned(*blk); delete blk; };
        auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        return torch::from_blob(blk->aligned, {(long long)padded}, deleter, opts);
    }

private:
    int open_with_direct_preference(const std::string& path, int flags, mode_t mode,
                                    bool direct_ok, bool strict_direct, bool& used_direct) {
        if (direct_ok) {
            used_direct = true;
            int fd = ::open(path.c_str(), flags | O_DIRECT, mode);
            if (fd >= 0) return fd;
            if (strict_direct) {
                throw std::runtime_error("O_DIRECT required but open failed: " + path + " errno=" + std::string(strerror(errno)));
            }
        }
        used_direct = false;
        return ::open(path.c_str(), flags, mode);
    }

#if AIMEMORY_WITH_URING
    void uring_read_exact(int fd, void* buf, size_t bytes, off_t off) {
        size_t done = 0;
        while (done < bytes) {
            io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
            throw_if(!sqe, "io_uring_get_sqe null");
            io_uring_prep_read(sqe, fd, (char*)buf + done, bytes - done, off + (off_t)done);

            int rc = io_uring_submit_and_wait(&ring_, 1);
            throw_if(rc < 0, "io_uring_submit_and_wait failed");

            io_uring_cqe* cqe = nullptr;
            rc = io_uring_wait_cqe(&ring_, &cqe);
            throw_if(rc < 0 || !cqe, "io_uring_wait_cqe failed");
            throw_if(cqe->res < 0, "io_uring read failed");
            throw_if(cqe->res == 0, "io_uring short read");
            done += (size_t)cqe->res;
            io_uring_cqe_seen(&ring_, cqe);
        }
    }

    void uring_write_exact(int fd, const void* buf, size_t bytes, off_t off) {
        size_t done = 0;
        while (done < bytes) {
            io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
            throw_if(!sqe, "io_uring_get_sqe null");
            io_uring_prep_write(sqe, fd, (char*)buf + done, bytes - done, off + (off_t)done);

            int rc = io_uring_submit_and_wait(&ring_, 1);
            throw_if(rc < 0, "io_uring_submit_and_wait failed");

            io_uring_cqe* cqe = nullptr;
            rc = io_uring_wait_cqe(&ring_, &cqe);
            throw_if(rc < 0 || !cqe, "io_uring_wait_cqe failed");
            throw_if(cqe->res < 0, "io_uring write failed");
            throw_if(cqe->res == 0, "io_uring short write");
            done += (size_t)cqe->res;
            io_uring_cqe_seen(&ring_, cqe);
        }
    }
#endif

public:
    void read_file_to_host_ptr(const std::string& path, uintptr_t hostPtr, size_t bytes, size_t offset, bool strict_direct=false) {
        if (bytes == 0) return;
        void* dst = (void*)hostPtr;
        bool direct_ok = is_aligned_u64((uint64_t)hostPtr,kAlign) && is_aligned_u64((uint64_t)offset,kAlign) && is_aligned_u64((uint64_t)bytes,kAlign);

        bool used_direct=false;
        int fd = open_with_direct_preference(path, O_RDONLY, 0, direct_ok, strict_direct, used_direct);
        throw_if(fd < 0, "open read failed: " + path + " errno=" + std::string(strerror(errno)));
        last_used_direct_.store(used_direct);

#if AIMEMORY_WITH_URING
        if (use_uring_) uring_read_exact(fd, dst, bytes, (off_t)offset);
        else pread_full(fd, dst, bytes, (off_t)offset);
#else
        pread_full(fd, dst, bytes, (off_t)offset);
#endif
        ::close(fd);
    }

    void write_host_ptr_to_file(const std::string& path, uintptr_t hostPtr, size_t bytes, size_t offset, bool strict_direct=false) {
        if (bytes == 0) return;
        const void* src = (const void*)hostPtr;
        bool direct_ok = is_aligned_u64((uint64_t)hostPtr,kAlign) && is_aligned_u64((uint64_t)offset,kAlign) && is_aligned_u64((uint64_t)bytes,kAlign);

        bool used_direct=false;
        int fd = open_with_direct_preference(path, O_WRONLY|O_CREAT, 0644, direct_ok, strict_direct, used_direct);
        throw_if(fd < 0, "open write failed: " + path + " errno=" + std::string(strerror(errno)));
        last_used_direct_.store(used_direct);

        ensure_file_size_delta_alloc(fd, (off_t)offset + (off_t)bytes);

#if AIMEMORY_WITH_URING
        if (use_uring_) uring_write_exact(fd, src, bytes, (off_t)offset);
        else pwrite_full(fd, src, bytes, (off_t)offset);
#else
        pwrite_full(fd, src, bytes, (off_t)offset);
#endif
        ::close(fd);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<SARCCore>(m, "SARCCore")
        .def(py::init<size_t>(), py::arg("staging_mb")=512)
        .def("gds_enabled", &SARCCore::gds_enabled)
        .def("uring_enabled", &SARCCore::uring_enabled)
        .def("last_used_direct", &SARCCore::last_used_direct)
        .def("staging_capacity_bytes", &SARCCore::staging_capacity_bytes)
        .def("staging_ptr", &SARCCore::staging_ptr)
        .def("round_up_4k", &SARCCore::round_up_4k)
        .def("alloc_pinned_u8", &SARCCore::alloc_pinned_u8)
        .def("read_file_to_host_ptr", &SARCCore::read_file_to_host_ptr)
        .def("write_host_ptr_to_file", &SARCCore::write_host_ptr_to_file);
}
