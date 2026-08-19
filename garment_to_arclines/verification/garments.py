"""Registry of which panels are currently verifiable per GarmentCodeData
garment, given what the loader currently supports.

Update this as loader.py grows to support more edge types.
"""

GARMENTS: dict[str, list[str]] = {
    # Skirt: all edges straight (step 1).
    "rand_2E2EL4UZUS": ["skirt_front", "skirt_back", "wb_front", "wb_back"],
    # Shirt: all edge types now supported (straight, circle, quadratic, cubic) -- full garment.
    "rand_2M8H83CQJP": [
        "right_ftorso", "right_btorso", "sl_right_cuff_f", "sl_right_cuff_b",
        "right_sleeve_b", "right_sleeve_f", "left_btorso", "sl_left_cuff_f",
        "sl_left_cuff_b", "left_sleeve_b", "left_sleeve_f", "left_ftorso",
        "wb_front", "wb_back",
    ],
}
