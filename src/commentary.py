"""
commentary.py — Maps event dicts → natural language commentary strings.
Uses randomized synonymous templates for variety.
"""

import random

# Template pools: {event_type: [template_strings]}
# Use {A}, {B} as player label placeholders.

TEMPLATES = {
    # ───── Basketball ─────
    "PASS": [
        "Player {A} passes the ball to Player {B}!",
        "Nice pass from Player {A} to Player {B}!",
        "{A} finds {B} with a clean pass!",
        "Player {A} dishes it off to Player {B}!",
    ],
    "SHOT": [
        "Player {A} goes for the shot!",
        "{A} attempts a shot at the basket!",
        "Player {A} launches it toward the hoop!",
        "{A} takes the shot — will it go in?",
    ],
    "DRIBBLE": [
        "Player {A} dribbling down the court!",
        "{A} drives with the ball!",
        "Player {A} handles the dribble smoothly!",
    ],
    "INTERCEPT": [
        "Player {A} intercepts the ball!",
        "{A} reading the play and cuts in!",
        "Player {A} steals possession!",
        "Turnover! {A} takes control!",
    ],

    # ───── Volleyball ─────
    "SERVE": [
        "Player {A} serves the ball!",
        "{A} steps up and delivers the serve!",
        "Player {A} starts the rally with a serve!",
        "Strong serve from Player {A}!",
    ],
    "SPIKE": [
        "Player {A} spikes it down hard!",
        "{A} with a powerful spike!",
        "Player {A} hammers the ball over the net!",
        "What a spike from Player {A}!",
    ],
    "SET": [
        "Player {A} sets the ball up beautifully!",
        "{A} with a perfect set!",
        "Great set by Player {A}!",
    ],
    "BLOCK": [
        "Player {A} goes up for the block!",
        "{A} walls it up at the net!",
        "Excellent block attempt by Player {A}!",
    ],
    "RALLY": [
        "The ball crosses the net — rally continues!",
        "Back and forth — the rally is on!",
        "Great exchange as the rally builds!",
    ],

    # ───── Football / Soccer ─────
    "SHOOT": [
        "Player {A} takes a shot on goal!",
        "{A} unleashes a strike!",
        "Player {A} goes for it — shooting!",
        "Powerful shot from Player {A}!",
    ],
    "TACKLE": [
        "Player {A} wins the tackle!",
        "{A} dispossesses the opponent!",
        "Crunching tackle from Player {A}!",
        "Player {A} clears the danger with a tackle!",
    ],
    "DRIBBLE": [
        "Player {A} dribbling past the opposition!",
        "{A} on the move with the ball!",
        "Great footwork from Player {A}!",
        "Player {A} takes on the defender!",
    ],
    "HEADER": [
        "Player {A} rises for the header!",
        "{A} meets the ball with a header!",
        "Headed on by Player {A}!",
    ],
    "FREE_KICK": [
        "Player {A} lines up for the free kick!",
        "{A} steps up to take the free kick!",
    ],

    # ───── Common ─────
    "SPRINT": [
        "Player {A} sprinting hard!",
        "{A} makes a powerful run!",
        "Player {A} bursting forward!",
    ],
    "FILLER": [
        "The intensity is palpable on the court right now!",
        "Both teams looking very focused in this phase of the match.",
        "A real tactical battle unfolding before our eyes.",
        "The energy from the players is phenomenal!",
        "Great atmosphere here as the game continues.",
        "Neither side willing to give an inch in this contest.",
        "Watching some truly disciplined play right now.",
        "The pace of the game is keeping everyone on their toes!",
    ],
    "MOVE": [
        "{A} is moving {DIR}.",
        "{A} shifts play {DIR}!",
        "Player {A} pushing {DIR}.",
    ],
    "DRIVE": [
        "{A} is driving hard to the basket!",
        "{A} attacking the paint!",
        "Dynamic drive from Player {A}!",
    ],
    "DEFEND": [
        "{A} is being closely marked by {B}.",
        "Tight defense from {B} on {A}!",
        "Player {B} matches {A} step for step.",
    ],
}


def generate(event: dict) -> str | None:
    """
    event = {'event': str, 'players': list[str], 'frame': int, 'dir': str (optional)}
    Returns a commentary string or None if event type unknown.
    """
    key = event.get("event", "")
    pls = event.get("players", [])
    direct = event.get("dir", "ahead")
    pool = TEMPLATES.get(key)
    if not pool:
        return None

    template = random.choice(pool)
    # Substitute labels for "A player" if they are "?" or None
    raw_A = pls[0] if pls else None
    raw_B = pls[1] if len(pls) > 1 else None

    # Determine display names for A and B
    A = f"Player {raw_A}" if (raw_A and raw_A != "?") else "A player"
    B = f"Player {raw_B}" if (raw_B and raw_B != "?") else "another player"

    # Handle both "Player {A}" and "{A}" formats consistently
    res = template.replace("Player {A}", "{A}").replace("Player {B}", "{B}")
    return res.format(A=A, B=B, DIR=direct)
