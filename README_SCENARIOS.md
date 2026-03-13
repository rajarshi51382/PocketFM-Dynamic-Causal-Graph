# Scenario Test Checklist

Use this list to validate the live demo. The intent is to ensure belief updates, causal propagation, and persistence behave as expected.

## ✅ Core Belief Polarization

1. Click **"🏰 Castle is unsafe"** 5–7 times.
2. Expected:
   - `castle_is_safe` log-odds decreases each time.
   - After repeated clicks, the value should cross below 0.

## ✅ Causal Cascade

1. After repeated **"Castle is unsafe"** clicks, observe `king_is_wise`.
2. Expected:
   - `king_is_wise` trends downward once `castle_is_safe` becomes negative.

## ✅ Preset Scenario Sanity

- **🏰 Castle is unsafe** → `castle_is_safe` decreases.
- **👑 King betrayed us** → `king_is_wise` decreases.
- **🌲 Forest now safe** → `forest_is_dangerous` decreases.
- **🤝 Ally is trustworthy** → relationship trust increases (if enabled).
- **⚔️ Enemy approaching** → valence decreases, arousal increases.
- **🕊️ Peace declared** → valence increases, arousal decreases.

## ✅ Save/Load Consistency

1. Click **Save**.
2. Modify beliefs via presets.
3. Click **Load**.
4. Expected:
   - Belief values return to the saved state.

## ✅ LLM Optional Path

1. Provide a Gemini API key in the sidebar.
2. Send a free-form prompt.
3. Expected:
   - Structured extraction and response work, no crashes.

---

If any expected outcome fails, capture the belief values and steps taken so we can reproduce quickly.