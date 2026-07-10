from engine import Room


def roman_amphitheater() -> tuple:
    room = Room()
    W, H = 40.0, 25.0

    room.add_wall(0,   0,  0,  H,  "marble")
    room.add_wall(0,   0,  W,  0,  "marble")
    room.add_wall(W,   0,  W,  H,  "stone")
    room.add_wall(W,   H,  0,  H,  "air_gap")
    room.add_wall(10,  0, 40, 18,  "stone")

    source   = ( 3.0, 12.0)
    receiver = (28.0, 12.0)
    desc = (
        "Roman Stone Theatre  |  40 m wide x 25 m deep  |  "
        "Marble scaena frons reflector, stone cavea seating banks, open sky"
    )
    return room, source, receiver, desc


def renaissance_cathedral() -> tuple:
    room = Room()
    W, H = 30.0, 18.0

    room.add_wall(0,  0,  W,  0,  "stone")
    room.add_wall(W,  0,  W,  H,  "stone")
    room.add_wall(W,  H,  0,  H,  "plaster")
    room.add_wall(0,  H,  0,  0,  "stone")
    room.add_wall(15, 0, 15,  8,  "tapestry")
    room.add_wall(20, 0, 20,  8,  "tapestry")

    source   = (25.0, 1.5)
    receiver = ( 8.0, 1.5)
    desc = (
        "Renaissance Cathedral Nave  |  30 m long x 18 m tall  |  "
        "Stone vaulting, plaster ceiling, heavy tapestry choir hangings"
    )
    return room, source, receiver, desc


def greek_bouleuterion() -> tuple:
    room = Room()
    W, H = 18.0, 10.0

    room.add_wall(0,  0,  W,  0,   "soil")
    room.add_wall(W,  0,  W,  H,   "stone")
    room.add_wall(W,  H,  0,  H,   "wood_panel")
    room.add_wall(0,  H,  0,  0,   "stone")
    room.add_wall(7,  0,  7,  1.5, "wood_panel")
    room.add_wall(11, 0, 11,  1.5, "wood_panel")

    source   = (9.0, 0.8)
    receiver = (9.0, 8.5)
    desc = (
        "Greek Bouleuterion  |  18 m x 10 m  |  "
        "Earth floor, stone bench walls, timber roof -- council debate hall"
    )
    return room, source, receiver, desc


def medieval_great_hall() -> tuple:
    room = Room()
    W, H = 25.0, 8.0

    room.add_wall(0, 0,  W,  0,  "wood_floor")
    room.add_wall(W, 0,  W,  H,  "stone")
    room.add_wall(W, H,  0,  H,  "wood_panel")
    room.add_wall(0, H,  0,  0,  "stone")
    for x in [5.0, 10.0, 15.0, 20.0]:
        room.add_wall(x,     0,  x + 1, 0,  "curtain")
        room.add_wall(x,     H,  x + 1, H,  "curtain")

    source   = (22.0, 1.5)
    receiver = (12.0, 1.5)
    desc = (
        "Medieval Castle Great Hall  |  25 m long x 8 m tall  |  "
        "Stone walls, oak ceiling, curtained window bays, wooden floor"
    )
    return room, source, receiver, desc


PRESETS: dict[str, callable] = {
    "amphitheater":  roman_amphitheater,
    "cathedral":     renaissance_cathedral,
    "bouleuterion":  greek_bouleuterion,
    "great_hall":    medieval_great_hall,
}
