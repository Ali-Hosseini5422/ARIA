"""Gas-accounting wrap. Does not re-allocate winners."""

GAS_PER_HIRE = 281_292


def gas_for_hires(n_hires: int) -> dict:
    return {
        "gas_per_hire": GAS_PER_HIRE,
        "gas_slot": GAS_PER_HIRE * int(n_hires),
        "mia_public": 0.917,
        "mia_rail": 0.679,
    }
