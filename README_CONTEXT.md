# Grounded Context (MVP)

This MVP is a **sandboxed fantasy scenario** used to validate dynamic causal character graphs. It is **not tied to a specific canon** or show timeline yet.

## Default Character

- **Name:** `Sir_Galahad`
- **Traits:** bravery 0.8, honesty 0.6, neuroticism 0.4, trusting 0.2

## Initial Beliefs

| Belief | Initial Log-Odds | Interpretation |
|---|---:|---|
| `castle_is_safe` | 1.5 | Strongly believes the castle is safe |
| `forest_is_dangerous` | 1.0 | Believes the forest is dangerous |
| `king_is_wise` | 0.5 | Mildly believes the king is wise |

## Causal Links

- `castle_is_safe` → `king_is_wise` (weight 0.8)
- `forest_is_dangerous` → `castle_is_safe` (weight 0.5)
- `not_castle_is_safe` → `not_king_is_wise` (weight 0.8)

## Grounding Assumptions

- **User statements are treated as direct observations** (credibility 1.0).
- **Negated propositions** (e.g., `not_castle_is_safe`) update the base belief inversely.
- **No external canon** is enforced; world constraints are optional.

## Future Canon Grounding (Planned)

To ground to a specific show or episode timeline, we can:
- Load **episode snapshots** as initial belief/relationship states.
- Enforce **timeline constraints** via `world_constraints`.
- Lock beliefs that should not change after a given time index.

If you want this, provide the narrative timeline and key facts to seed the state.