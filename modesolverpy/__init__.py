from modesolverpy.config import CONFIG


def get_modes_jsonpath(mode_solver):
    return CONFIG["cache"] / f"{mode_solver.name}.json"


__all__ = ["CONFIG", "get_modes_jsonpath"]

if __name__ == "__main__":
    print(__all__)
