import torch

import arc1


def main():
    ctrl0 = arc1.enable(backend="NOOP")
    ctrl1 = arc1.enable(backend="NOOP")
    assert ctrl0 is ctrl1

    model = torch.nn.Linear(8, 4)
    x = torch.randn(2, 8)
    y = model(x).sum()
    y.backward()

    st = arc1.status()
    assert bool(st.get("enabled", False))
    assert bool(st.get("has_controller", False))

    arc1.disable(shutdown=True)
    st2 = arc1.status()
    assert not bool(st2.get("enabled", True))
    print("ARC1_ONE_LINE_OK")


if __name__ == "__main__":
    main()
