LIP_LABELS = [
    "background",
    "hat",
    "hair",
    "glove",
    "sunglasses",
    "upper_clothes",
    "dress",
    "coat",
    "socks",
    "pants",
    "jumpsuits",
    "scarf",
    "skirt",
    "face",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "left_shoe",
    "right_shoe",
]

LIP_NAME_TO_ID = {name: idx for idx, name in enumerate(LIP_LABELS)}

# Common aliases produced by VLMs are normalized into LIP label ids.
LIP_ALIASES = {
    "bg": "background",
    "cap": "hat",
    "headwear": "hat",
    "hair": "hair",
    "gloves": "glove",
    "glove": "glove",
    "glasses": "sunglasses",
    "sunglass": "sunglasses",
    "sunglasses": "sunglasses",
    "top": "upper_clothes",
    "shirt": "upper_clothes",
    "tshirt": "upper_clothes",
    "t-shirt": "upper_clothes",
    "blouse": "upper_clothes",
    "upper-clothes": "upper_clothes",
    "upper clothes": "upper_clothes",
    "upper_clothes": "upper_clothes",
    "dress": "dress",
    "coat": "coat",
    "sock": "socks",
    "socks": "socks",
    "trousers": "pants",
    "pant": "pants",
    "pants": "pants",
    "jumpsuit": "jumpsuits",
    "jumpsuits": "jumpsuits",
    "scarf": "scarf",
    "skirt": "skirt",
    "face": "face",
    "left arm": "left_arm",
    "l_arm": "left_arm",
    "left_arm": "left_arm",
    "right arm": "right_arm",
    "r_arm": "right_arm",
    "right_arm": "right_arm",
    "left leg": "left_leg",
    "l_leg": "left_leg",
    "left_leg": "left_leg",
    "right leg": "right_leg",
    "r_leg": "right_leg",
    "right_leg": "right_leg",
    "left shoe": "left_shoe",
    "left_shoe": "left_shoe",
    "left boot": "left_shoe",
    "right shoe": "right_shoe",
    "right_shoe": "right_shoe",
    "right boot": "right_shoe",
}


def normalize_lip_label(label: str):
    if not isinstance(label, str):
        return None
    key = label.strip().lower().replace("-", " ").replace("/", " ")
    key = "_".join(key.split())
    mapped = LIP_ALIASES.get(key, key)
    return mapped if mapped in LIP_NAME_TO_ID else None
