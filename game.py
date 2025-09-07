import requests
import uuid
import time
import argparse
import os
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Base URL of the API
BASE_URL = "https://berghain.challenges.listenlabs.ai"

def http_get_with_retry(url, params=None, retries: int = 3, base_backoff_seconds: float = 0.5, timeout_seconds: float = 20.0):
    """
    Lightweight GET with exponential backoff to avoid transient failures / stalls.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return requests.get(url, params=params, timeout=timeout_seconds)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                sleep_seconds = base_backoff_seconds * (2 ** attempt)
                time.sleep(sleep_seconds)
            else:
                raise

def create_new_game(scenario, player_id):
    """
    Creates a new game and returns the game details.

    Args:
        scenario: The game scenario (1, 2, or 3).
        player_id: The player's ID.

    Returns:
        A dictionary containing gameId, constraints, and attributeStatistics.
    """
    endpoint = f"{BASE_URL}/new-game"
    params = {"scenario": scenario, "playerId": player_id}
    response = http_get_with_retry(endpoint, params=params)
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.json()

def decide_and_next(game_id, person_index, accept=None):
    """
    Sends a decision for the current person and gets the next person's attributes.

    Args:
        game_id: The ID of the current game.
        person_index: The index of the current person.
        accept: True to accept the person, False to reject.

    Returns:
        A dictionary containing the next person's attributes or a game over message.
    """
    endpoint = f"{BASE_URL}/decide-and-next"
    # Only include 'accept' for decisions after the first fetch (personIndex=0 without decision)
    params = {"gameId": game_id, "personIndex": person_index}
    if accept is not None:
        # API expects lowercase 'true'/'false' in query, avoid Python True/False
        params["accept"] = "true" if bool(accept) else "false"
    response = http_get_with_retry(endpoint, params=params)
    response.raise_for_status() # Raise an exception for bad status codes
    return response.json()

def play_game(scenario, player_id, decision_function):
    """
    Plays a game of the Berghain challenge using a given decision function.

    Args:
        scenario: The game scenario (1, 2, or 3).
        player_id: The player's ID.
        decision_function: A function that takes current_person_attributes,
                           constraints, accepted_attributes_count,
                           accepted_count, rejected_count, and venue_capacity
                           as input and returns True to accept or False to reject.

    Returns:
        A dictionary containing the final accepted_count, rejected_count,
        and accepted_attributes_count, or an error message.
    """
    try:
        game_details = create_new_game(scenario, player_id)
        game_id = game_details["gameId"]
        constraints = game_details["constraints"]
        attribute_statistics = game_details["attributeStatistics"]

        print(f"Starting game with ID: {game_id}")
        print("Constraints:", constraints)
        print("Attribute Statistics:", attribute_statistics)

        accepted_count = 0
        rejected_count = 0
        current_person_index = 0
        venue_capacity = 1000
        rejection_limit = 20000

        accepted_attributes_count = {constraint['attribute']: 0 for constraint in constraints}

        # Fetch the first person (personIndex=0) WITHOUT making a blind decision
        try:
            next_person_data = decide_and_next(game_id, current_person_index, accept=None)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching initial person 0: {e}")
            return {"error": f"Error during initial fetch: {e}"}


        # Now loop through the rest of the people
        while accepted_count < venue_capacity and rejected_count < rejection_limit:
            status = next_person_data.get("status")
            if status == "completed":
                print("Game completed by server. Stopping loop.")
                break
            if status == "failed":
                print("Game failed:", next_person_data.get("reason"))
                break

            person_to_decide_attributes = next_person_data.get("nextPerson", {}).get("attributes", {})
            person_to_decide_index = next_person_data.get("nextPerson", {}).get("personIndex")

            if person_to_decide_index is None:
                print("Error: Could not get next person's index. Ending game loop.")
                break

            # Use the provided decision function
            accept_person = decision_function(
                person_to_decide_attributes,
                constraints,
                accepted_attributes_count,
                accepted_count,
                rejected_count,
                venue_capacity,
                attribute_statistics
            )

            try:
                # Submit decision for the current person and receive the next person
                next_person_data = decide_and_next(game_id, person_to_decide_index, accept_person)

                if accept_person:
                    accepted_count += 1
                    for attribute, value in person_to_decide_attributes.items():
                        if value and attribute in accepted_attributes_count:
                            accepted_attributes_count[attribute] += 1
                else:
                    rejected_count += 1

            except requests.exceptions.RequestException as e:
                print(f"Error during decision for person {person_to_decide_index}: {e}")
                # Potentially handle specific errors like rate limits here
                return {"error": f"Error during game progression: {e}"}

        print(f"Game finished. Accepted: {accepted_count}, Rejected: {rejected_count}")
        print(f"Final accepted attribute counts: {accepted_attributes_count}")

        return {
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "accepted_attributes_count": accepted_attributes_count
        }

    except requests.exceptions.RequestException as e:
        print(f"Error creating new game: {e}")
        # Potentially handle rate limit for new game creation here
        return {"error": f"Error creating new game: {e}"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


def deficit_aware_decision(current_person_attributes, constraints, accepted_attributes_count,
                           accepted_count, rejected_count, venue_capacity, attribute_statistics,
                           safety_z: float = 0.0):
    """
    Accept if doing so keeps feasibility for all min-count constraints and ideally reduces deficits.

    Strategy:
    - Compute per-attribute deficits d_i = max(0, minCount_i - accepted_count_i)
    - Remaining capacity R = venue_capacity - accepted_count
    - Must-reject if exists i with (d_i > R - 1 and current_person_attributes.get(i) == False)
    - Otherwise:
        - If candidate helps any unmet constraint (has i with d_i > 0 and attr True): accept
        - Else (neutral), accept only if slack_lower_bound = R - sum(d_i) > 0; otherwise reject
    """
    # Build quick maps for min counts
    min_counts = {c["attribute"]: c["minCount"] for c in constraints}
    # Ensure all constrained attributes tracked
    deficits = {}
    for attr, need in min_counts.items():
        have = accepted_attributes_count.get(attr, 0)
        deficits[attr] = max(0, need - have)

    R = venue_capacity - accepted_count
    # Safety: if R <= 0, we should not accept anyone
    if R <= 0:
        return False

    # Frequencies for expected future supply
    rel_freq = (attribute_statistics or {}).get("relativeFrequencies", {})

    accept_safe = is_decision_safe(
        current_person_attributes, deficits, rel_freq, R, safety_z, accept_decision=True
    )
    reject_safe = is_decision_safe(
        current_person_attributes, deficits, rel_freq, R, safety_z, accept_decision=False
    )
    if not accept_safe and reject_safe:
        return False

    # If rejecting breaks feasibility but accepting does not, accept
    if accept_safe and not reject_safe:
        return True

    # If both are safe (or both unsafe), fall back to deficit and slack heuristics
    # Must-reject test: accepting a non-attr person would make any constraint infeasible by capacity alone
    for attr, d in deficits.items():
        if d > R - 1 and not current_person_attributes.get(attr, False):
            return False

    # Helpful if person satisfies any still-unmet constraint
    helpful = any(current_person_attributes.get(attr, False) and d > 0 for attr, d in deficits.items())
    if helpful:
        return True

    # Neutral: consume slack only if there is guaranteed slack (lower-bound via sum of deficits)
    sum_deficits = sum(deficits.values())
    slack_lower_bound = R - sum_deficits
    return slack_lower_bound > 0


def is_decision_safe(current_person_attributes, deficits, rel_freq, remaining_capacity, safety_z, accept_decision: bool) -> bool:
    """
    Evaluate whether accept/reject keeps feasibility likely, using conservative expected supply.

    If accept_decision is True, we assume the current person consumes one capacity and contributes
    +1 to any attribute they have. If False, capacity remains the same and no contribution is made.
    """
    if accept_decision:
        remaining_capacity_after_accept = remaining_capacity - 1
        if remaining_capacity_after_accept < 0:
            return False
        for attr, d in deficits.items():
            contribution = 1 if current_person_attributes.get(attr, False) else 0
            remaining_deficit = max(0, d - contribution)
            p = float(rel_freq.get(attr, 0.0))
            expected_supply = p * remaining_capacity_after_accept
            variance = p * (1.0 - p) * max(remaining_capacity_after_accept, 0)
            std_dev = variance ** 0.5
            conservative_supply = expected_supply - safety_z * std_dev
            if conservative_supply + 1e-9 < remaining_deficit:
                return False
        return True
    else:
        # Reject decision: capacity unchanged; no contribution
        for attr, d in deficits.items():
            p = float(rel_freq.get(attr, 0.0))
            expected_supply = p * remaining_capacity
            variance = p * (1.0 - p) * max(remaining_capacity, 0)
            std_dev = variance ** 0.5
            conservative_supply = expected_supply - safety_z * std_dev
            if conservative_supply + 1e-9 < d:
                return False
        return True


def llm_decision_with_gating(current_person_attributes, constraints, accepted_attributes_count,
                             accepted_count, rejected_count, venue_capacity, attribute_statistics,
                             safety_z: float = 0.5, provider: str = "gemini",
                             model: str = "gemini-1.5-flash", temperature: float = 0.2,
                             api_key: str = None):
    """
    LLM-driven decision with feasibility gating:
    - If only one of accept/reject is safe, enforce that action.
    - If both unsafe, fall back to deficit/slug heuristic (no LLM call).
    - If both safe, call the LLM with a compact prompt and parse ACCEPT/REJECT.
    """
    # Compute deficits
    min_counts = {c["attribute"]: c["minCount"] for c in constraints}
    deficits = {attr: max(0, min_counts[attr] - accepted_attributes_count.get(attr, 0)) for attr in min_counts}
    R = venue_capacity - accepted_count
    if R <= 0:
        return False

    rel_freq = (attribute_statistics or {}).get("relativeFrequencies", {})
    accept_safe = is_decision_safe(current_person_attributes, deficits, rel_freq, R, safety_z, True)
    reject_safe = is_decision_safe(current_person_attributes, deficits, rel_freq, R, safety_z, False)

    if not accept_safe and reject_safe:
        return False
    if accept_safe and not reject_safe:
        return True

    # If both unsafe, rely on deterministic heuristic rather than LLM
    if not accept_safe and not reject_safe:
        helpful = any(current_person_attributes.get(attr, False) and d > 0 for attr, d in deficits.items())
        if helpful:
            return True
        sum_deficits = sum(deficits.values())
        return (R - sum_deficits) > 0

    # Both safe: query LLM
    # Build concise prompt
    helpful_attrs = [attr for attr, d in deficits.items() if d > 0 and current_person_attributes.get(attr, False)]
    current_true_attrs = [attr for attr, val in current_person_attributes.items() if val]
    constraints_lines = []
    for attr, need in min_counts.items():
        have = accepted_attributes_count.get(attr, 0)
        d = deficits.get(attr, 0)
        p = rel_freq.get(attr, 0.0)
        constraints_lines.append(f"- {attr}: min={need}, have={have}, deficit={d}, p={p}")

    prompt = (
        "You are optimizing decisions for a sequential admission game. "
        "Return strictly 'ACCEPT' or 'REJECT' (uppercase, no punctuation).\n\n"
        f"Accepted={accepted_count}, Rejected={rejected_count}, RemainingCapacity={R}.\n"
        f"Current person true attributes: {current_true_attrs}.\n"
        f"Helpful attributes (reduce deficits): {helpful_attrs}.\n"
        "Constraints and counts:\n" + "\n".join(constraints_lines) + "\n"
        "Decision rules: prefer actions that reduce deficits; keep feasibility w.r.t. min counts; "
        "break ties by accepting neutrals only if slack (remaining capacity minus sum of deficits) is positive.\n"
        "Answer with exactly one token: ACCEPT or REJECT."
    )

    try:
        if provider == "gemini":
            # Lazy import to avoid hard dependency
            import google.generativeai as genai  # type: ignore

            key = api_key or os.getenv("GEMINI_API_KEY")
            if not key:
                # Fallback if no key: deterministic heuristic
                helpful = any(current_person_attributes.get(attr, False) and d > 0 for attr, d in deficits.items())
                if helpful:
                    return True
                sum_deficits = sum(deficits.values())
                return (R - sum_deficits) > 0

            genai.configure(api_key=key)
            llm = genai.GenerativeModel(model)
            resp = llm.generate_content(prompt, generation_config={
                "temperature": float(temperature),
                "max_output_tokens": 8,
            })
            text = (getattr(resp, "text", None) or "").strip().upper()
            if "ACCEPT" in text and "REJECT" not in text:
                return True
            if "REJECT" in text and "ACCEPT" not in text:
                return False
            # If ambiguous, use helpful/slack fallback
            helpful = any(current_person_attributes.get(attr, False) and d > 0 for attr, d in deficits.items())
            if helpful:
                return True
            sum_deficits = sum(deficits.values())
            return (R - sum_deficits) > 0
        elif provider == "openrouter":
            key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API") or os.getenv("OPEN_ROUTER_API_KEY")
            # Decide model: CLI arg wins, else env, else a reasonable default
            model_to_use = model if model and model != "gemini-1.5-flash" else os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
            if not key:
                # Fallback if no key: deterministic heuristic
                helpful = any(current_person_attributes.get(attr, False) and d > 0 for attr, d in deficits.items())
                if helpful:
                    return True
                sum_deficits = sum(deficits.values())
                return (R - sum_deficits) > 0

            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_to_use,
                "messages": [
                    {"role": "system", "content": "Return strictly 'ACCEPT' or 'REJECT' (uppercase, no punctuation)."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": float(temperature),
                "max_tokens": 8,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            try:
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
            except Exception:
                text = ""
            if "ACCEPT" in text and "REJECT" not in text:
                return True
            if "REJECT" in text and "ACCEPT" not in text:
                return False
            # If ambiguous, use helpful/slack fallback
            helpful = any(current_person_attributes.get(attr, False) and d > 0 for attr, d in deficits.items())
            if helpful:
                return True
            sum_deficits = sum(deficits.values())
            return (R - sum_deficits) > 0
        else:
            # Unsupported provider: fallback
            helpful = any(current_person_attributes.get(attr, False) and d > 0 for attr, d in deficits.items())
            if helpful:
                return True
            sum_deficits = sum(deficits.values())
            return (R - sum_deficits) > 0
    except Exception:
        # Any LLM error -> safe fallback
        helpful = any(current_person_attributes.get(attr, False) and d > 0 for attr, d in deficits.items())
        if helpful:
            return True
        sum_deficits = sum(deficits.values())
        return (R - sum_deficits) > 0


def intersection_decision(current_person_attributes, constraints, accepted_attributes_count,
                          accepted_count, rejected_count, venue_capacity, attribute_statistics,
                          safety_z: float = 0.0):
    """
    Binding-constraint intersection policy:
    - Identify currently unmet attributes (deficit > 0).
    - Accept only if the person has ALL of the unmet attributes (intersection).
    - If no deficits remain, accept all to fill capacity quickly.
    - Otherwise reject.
    """
    min_counts = {c["attribute"]: c["minCount"] for c in constraints}
    deficits = {attr: max(0, min_counts[attr] - accepted_attributes_count.get(attr, 0)) for attr in min_counts}
    if sum(deficits.values()) == 0:
        return True
    unmet = [attr for attr, d in deficits.items() if d > 0]
    if not unmet:
        return True
    # Accept only if person satisfies all unmet attributes
    return all(current_person_attributes.get(attr, False) for attr in unmet)


def threshold_decision(current_person_attributes, constraints, accepted_attributes_count,
                       accepted_count, rejected_count, venue_capacity, attribute_statistics,
                       helpful_threshold: float = 0.5, alpha: float = 1.5):
    """
    Weighted-threshold strategy for multi-constraint scenarios:
    - Score each person by how much they reduce weighted deficits.
    - Weight deficit of attribute i by w_i = (deficit_i) ^ alpha, optionally scaled by 1/p_i.
    - Accept if normalized helpfulness >= helpful_threshold.
    - If no active deficits (all satisfied), accept to fill capacity.
    """
    rel_freq = (attribute_statistics or {}).get("relativeFrequencies", {})
    min_counts = {c["attribute"]: c["minCount"] for c in constraints}
    deficits = {attr: max(0, min_counts[attr] - accepted_attributes_count.get(attr, 0)) for attr in min_counts}
    total_deficit = sum(deficits.values())
    if total_deficit == 0:
        return True

    # Compute weights emphasizing hard-to-meet constraints
    weights = {}
    for attr, d in deficits.items():
        if d <= 0:
            continue
        p = float(rel_freq.get(attr, 0.0))
        # Emphasize higher deficits and rarer attributes
        base = (d ** float(alpha))
        rarity_boost = (1.0 / max(p, 1e-6))
        weights[attr] = base * rarity_boost

    if not weights:
        return True

    # Person helpfulness is sum of weights for attributes they satisfy
    helpful_score = sum(weight for attr, weight in weights.items() if current_person_attributes.get(attr, False))
    max_possible = sum(weights.values())
    if max_possible <= 0:
        return False
    helpfulness = helpful_score / max_possible

    return helpfulness >= float(helpful_threshold)


def chernoff_required_buffer(remaining_capacity: int, delta: float) -> float:
    # sqrt(0.5 * R * log(1/delta))
    import math
    return math.sqrt(max(0.0, 0.5 * float(remaining_capacity) * math.log(max(1.0, 1.0 / max(delta, 1e-9)))))


def primal_dual_decision(current_person_attributes, constraints, accepted_attributes_count,
                         accepted_count, rejected_count, venue_capacity, attribute_statistics,
                         base_alpha: float = 1.8, early_threshold: float = 0.65,
                         mid_threshold: float = 0.5, late_threshold: float = 0.4,
                         delta_early: float = 0.10, delta_mid: float = 0.05, delta_late: float = 0.02,
                         corr_penalty: float = 0.6):
    """
    Primal–dual decision with correlation-aware scoring and concentration-bounded feasibility.
    - Compute deficits and rarity-aware dual weights.
    - Phase thresholds by remaining capacity.
    - Penalize negative-correlation conflicts relative to unmet constraints; boost positive.
    - Enforce feasibility using Chernoff buffer.
    """
    import math

    rel_freq = (attribute_statistics or {}).get("relativeFrequencies", {})
    corrs = (attribute_statistics or {}).get("correlations", {})
    min_counts = {c["attribute"]: c["minCount"] for c in constraints}
    deficits = {attr: max(0, min_counts[attr] - accepted_attributes_count.get(attr, 0)) for attr in min_counts}

    R = int(max(0, venue_capacity - accepted_count))
    if R <= 0:
        return False

    # Phase scheduling
    if R > 700:
        helpful_threshold = early_threshold
        alpha = base_alpha
        delta = delta_early
    elif R > 300:
        helpful_threshold = mid_threshold
        alpha = max(1.2, base_alpha * 0.9)
        delta = delta_mid
    else:
        helpful_threshold = late_threshold
        alpha = max(1.0, base_alpha * 0.8)
        delta = delta_late

    # If all deficits met, accept to fill capacity quickly
    if sum(deficits.values()) == 0:
        return True

    # Feasibility guard with Chernoff buffer
    buffer = chernoff_required_buffer(R, delta)

    # Identify critical attributes (tight under buffer)
    critical_attrs = []
    for attr, d in deficits.items():
        if d <= 0:
            continue
        p = float(rel_freq.get(attr, 0.0))
        expected_after = p * max(R - 1, 0)
        if expected_after + buffer + 1e-9 < float(d):
            critical_attrs.append(attr)

    # Early feasibility gating: if there are critical attrs, require candidate to help at least one
    if critical_attrs:
        helps_critical = any(current_person_attributes.get(a, False) for a in critical_attrs)
        if not helps_critical:
            return False
        # If they help at least one critical attr, allow without further gating
        # (still score-based selection can accept immediately below)

    # Weighted dual prices with rarity boost
    weights = {}
    for attr, d in deficits.items():
        if d <= 0:
            continue
        p = float(rel_freq.get(attr, 0.0))
        rarity = 1.0 / max(p, 1e-6)
        weights[attr] = (float(d) ** float(alpha)) * rarity

    # Correlation adjustment: reward positive corr with unmet, penalize negative
    corr_adjust = 0.0
    for a, d_a in deficits.items():
        if d_a <= 0:
            continue
        for b, d_b in deficits.items():
            if b == a or d_b <= 0:
                continue
            rho = float(((corrs.get(a, {}) or {}).get(b, 0.0)))
            if current_person_attributes.get(a, False) and current_person_attributes.get(b, False):
                if rho < 0:
                    corr_adjust -= corr_penalty * abs(rho)
                elif rho > 0:
                    corr_adjust += 0.25 * rho

    helpful_score = sum(weight for attr, weight in weights.items() if current_person_attributes.get(attr, False))
    max_possible = sum(weights.values()) or 1.0
    helpfulness = (helpful_score / max_possible) + corr_adjust

    # Accept if helpful enough; else, allow neutral if slack exists
    if helpfulness >= helpful_threshold:
        return True

    # Slack check: if R - sum(deficits) > 0, allow neutrals occasionally to build buffer
    if (R - sum(deficits.values())) > 0 and helpfulness >= (helpful_threshold * 0.6):
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Play the Berghain Challenge with configurable strategies.")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3], help="Scenario number (1, 2, or 3)")
    parser.add_argument("--player-id", type=str, default=None, help="Player ID (UUID). If omitted, a new one is generated.")
    parser.add_argument("--safety", type=float, default=0.5, help="Safety z-score for conservative expected supply (e.g., 0.0-2.0)")
    parser.add_argument("--strategy", type=str, default="intersection", choices=["intersection", "deficit", "threshold", "primaldual", "llm", "auto"], help="Decision strategy")
    parser.add_argument("--helpful-threshold", type=float, default=0.5, help="Threshold helpfulness for threshold strategy (0-1)")
    parser.add_argument("--alpha", type=float, default=1.5, help="Exponent for deficit weighting in threshold strategy")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM-guided decisions with feasibility gating")
    parser.add_argument("--llm-provider", type=str, default="gemini", choices=["gemini", "openrouter"], help="LLM provider")
    parser.add_argument("--llm-model", type=str, default="gemini-1.5-flash", help="LLM model name")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--llm-api-key", type=str, default=None, help="LLM API key (or set GEMINI_API_KEY)")
    # Primal-dual CLI knobs
    parser.add_argument("--pd-base-alpha", type=float, default=1.8, help="Primal-dual base alpha for deficit weighting")
    parser.add_argument("--pd-early-threshold", type=float, default=0.65, help="Primal-dual early phase helpful threshold")
    parser.add_argument("--pd-mid-threshold", type=float, default=0.5, help="Primal-dual mid phase helpful threshold")
    parser.add_argument("--pd-late-threshold", type=float, default=0.4, help="Primal-dual late phase helpful threshold")
    parser.add_argument("--pd-delta-early", type=float, default=0.10, help="Chernoff delta for early phase")
    parser.add_argument("--pd-delta-mid", type=float, default=0.05, help="Chernoff delta for mid phase")
    parser.add_argument("--pd-delta-late", type=float, default=0.02, help="Chernoff delta for late phase")
    parser.add_argument("--pd-corr-penalty", type=float, default=0.6, help="Penalty for negative correlations in primal-dual")
    args = parser.parse_args()

    # Allow default from environment .env: PLAYER_ID
    player_id = args.player_id or os.getenv("PLAYER_ID") or str(uuid.uuid4())
    print(f"Using player_id: {player_id}")
    # Choose decision function
    if args.strategy == "llm" or args.use_llm:
        def decision_fn(attrs, constraints, counts, acc, rej, cap, stats):
            return llm_decision_with_gating(
                attrs, constraints, counts, acc, rej, cap, stats,
                safety_z=args.safety,
                provider=args.llm_provider,
                model=args.llm_model,
                temperature=args.llm_temperature,
                api_key=args.llm_api_key,
            )
    elif args.strategy == "deficit":
        def decision_fn(attrs, constraints, counts, acc, rej, cap, stats):
            return deficit_aware_decision(attrs, constraints, counts, acc, rej, cap, stats, safety_z=args.safety)
    elif args.strategy == "threshold":
        def decision_fn(attrs, constraints, counts, acc, rej, cap, stats):
            return threshold_decision(
                attrs, constraints, counts, acc, rej, cap, stats,
                helpful_threshold=args.helpful_threshold,
                alpha=args.alpha,
            )
    elif args.strategy == "primaldual":
        def decision_fn(attrs, constraints, counts, acc, rej, cap, stats):
            return primal_dual_decision(
                attrs, constraints, counts, acc, rej, cap, stats,
                base_alpha=args.pd_base_alpha,
                early_threshold=args.pd_early_threshold,
                mid_threshold=args.pd_mid_threshold,
                late_threshold=args.pd_late_threshold,
                delta_early=args.pd_delta_early,
                delta_mid=args.pd_delta_mid,
                delta_late=args.pd_delta_late,
                corr_penalty=args.pd_corr_penalty,
            )
    elif args.strategy == "auto":
        # Meta-strategy: early primal-dual, mid threshold, late deficit
        def decision_fn(attrs, constraints, counts, acc, rej, cap, stats):
            remaining = cap - acc
            if remaining > 700:
                # Scenario 3 often suffers with overly strict early gating; prefer threshold early
                return threshold_decision(
                    attrs, constraints, counts, acc, rej, cap, stats,
                    helpful_threshold=max(0.55, float(args.helpful_threshold)),
                    alpha=max(1.5, float(args.alpha)),
                )
            elif remaining > 300:
                return threshold_decision(
                    attrs, constraints, counts, acc, rej, cap, stats,
                    helpful_threshold=max(0.6, float(args.helpful_threshold)),
                    alpha=max(1.6, float(args.alpha)),
                )
            else:
                return deficit_aware_decision(
                    attrs, constraints, counts, acc, rej, cap, stats,
                    safety_z=max(0.2, float(args.safety))
                )
    else:
        def decision_fn(attrs, constraints, counts, acc, rej, cap, stats):
            return intersection_decision(attrs, constraints, counts, acc, rej, cap, stats, safety_z=args.safety)

    result = play_game(args.scenario, player_id, decision_fn)
    print("Result:", result)

    # Append a brief run summary to docs/v01_results.md
    try:
        summary_path = os.path.join(os.path.dirname(__file__), "docs", "v01_results.md")
        # Fallback if relative join fails (when running from repo root this works)
        if not os.path.exists(summary_path):
            summary_path = os.path.abspath(os.path.join(os.getcwd(), "docs", "v01_results.md"))
        lines = []
        lines.append(f"\n### Scenario {args.scenario} run summary")
        lines.append(f"- **strategy**: {args.strategy}")
        lines.append(f"- **accepted**: {result.get('accepted_count')}")
        lines.append(f"- **rejected**: {result.get('rejected_count')}")
        acc_counts = result.get("accepted_attributes_count", {}) or {}
        if acc_counts:
            # Render key counts on one line to keep it compact
            kv = ", ".join([f"{k}:{v}" for k, v in acc_counts.items()])
            lines.append(f"- **accepted_attributes_count**: {kv}")
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"Warning: could not append run summary: {e}")


if __name__ == "__main__":
    main()