"""
Metaphor / Literal Sentence Pairs
====================================

Curated dataset for testing whether metaphorical language exhibits
different gauge curvature than literal language.

Design principles:
    - Each pair shares the same key phrase, used metaphorically in one
      sentence and literally in the other.
    - Metaphors are PARTIALLY compositional: "drowning in work" maps
      the physical concept of drowning to feeling overwhelmed. The
      meaning is derivable through conceptual mapping, unlike idioms
      which are fully frozen.
    - This positions metaphor BETWEEN idioms and literal language on
      the compositionality gradient:
        literal (full) > metaphor (partial) > idiom (none)
    - Controls: complex descriptive language that is fully compositional.

Metaphor types included:
    - Conceptual metaphors (ARGUMENT IS WAR, TIME IS MONEY, etc.)
    - Sensory-to-abstract mappings (sharp mind, warm reception)
    - Physical-to-emotional mappings (heavy heart, crushed spirit)
    - Spatial metaphors (high hopes, deep understanding)

Labels:
    'metaphorical' — figurative/analogical meaning (medium expected curvature)
    'literal'      — concrete/physical meaning of same phrase (low expected curvature)
    'control'      — complex but fully compositional language
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MetaphorPair:
    """A labeled sentence for metaphor holonomy measurement."""
    text: str
    label: str          # 'metaphorical', 'literal', or 'control'
    pair_id: int        # links metaphorical/literal pairs
    metaphor: str = ''  # the shared metaphorical phrase (if applicable)


def load_metaphor_pairs() -> List[MetaphorPair]:
    """
    Load curated metaphor/literal/control dataset.

    Returns list of MetaphorPair with balanced labels.
    """
    pairs = []
    pid = 0

    # =====================================================================
    # PAIRED: same phrase, metaphorical vs literal usage
    # Each tuple: (metaphorical_sentence, literal_sentence, shared_phrase)
    # The phrase works both figuratively and literally.
    # =====================================================================

    _paired = [
        # --- Temperature / emotion mappings ---
        (
            "After years of feuding, the ice between them finally melted.",
            "After sitting in the sun, the ice in the glass finally melted.",
            "ice between them finally melted",
        ),
        (
            "She gave him a cold reception when he arrived at the party.",
            "The cold reception area had no heating and everyone shivered.",
            "cold reception",
        ),
        (
            "His warm personality made everyone feel at ease instantly.",
            "His warm jacket kept him comfortable in the freezing weather.",
            "warm",
        ),
        (
            "The debate was heated and neither side would back down.",
            "The soup was heated until it was too hot to eat.",
            "heated",
        ),
        (
            "She was burning with anger after reading the message.",
            "She was burning from the sunburn after falling asleep outside.",
            "burning",
        ),

        # --- Weight / pressure mappings ---
        (
            "He carried the weight of his past failures everywhere he went.",
            "He carried the weight of the heavy boxes up three flights of stairs.",
            "carried the weight",
        ),
        (
            "The heavy burden of responsibility fell on her shoulders.",
            "The heavy burden of firewood fell off the back of the truck.",
            "heavy burden",
        ),
        (
            "She felt crushed by the pressure of everyone's expectations.",
            "The car was crushed by the falling tree during the storm.",
            "crushed",
        ),
        (
            "He was drowning in work and could not take a break.",
            "He was drowning in the deep end and called for the lifeguard.",
            "drowning in",
        ),
        (
            "The weight of her grief made it hard to get out of bed.",
            "The weight of the stone made it impossible to lift alone.",
            "weight of",
        ),

        # --- Light / darkness mappings ---
        (
            "Her smile lit up the entire room when she walked in.",
            "The new chandelier lit up the entire room with bright light.",
            "lit up the entire room",
        ),
        (
            "He was in a dark place after losing his job and his apartment.",
            "He was in a dark place with no windows deep inside the building.",
            "dark place",
        ),
        (
            "A bright idea suddenly came to her during the meeting.",
            "A bright light suddenly came through the window at dawn.",
            "bright",
        ),
        (
            "The future looked dim after the company announced layoffs.",
            "The hallway looked dim because half the bulbs had burned out.",
            "dim",
        ),
        (
            "She cast a shadow over the celebration with her grim news.",
            "The tall building cast a shadow over the park all afternoon.",
            "cast a shadow over",
        ),

        # --- Sharpness / perception mappings ---
        (
            "He has a sharp mind and can solve problems very quickly.",
            "He has a sharp knife and can cut through anything easily.",
            "sharp",
        ),
        (
            "Her cutting remarks left everyone at the table speechless.",
            "The cutting wind left everyone on the hillside shivering.",
            "cutting",
        ),
        (
            "His pointed criticism made the presenter visibly uncomfortable.",
            "His pointed stick made a useful tool for roasting marshmallows.",
            "pointed",
        ),
        (
            "She has a keen eye for detail in her architectural designs.",
            "The hawk has a keen eye and spotted the mouse from high above.",
            "keen eye",
        ),

        # --- Spatial / journey mappings ---
        (
            "She was at a crossroads in her career and had to decide.",
            "She was at a crossroads in the trail and checked her map.",
            "at a crossroads",
        ),
        (
            "His life was going in circles and nothing ever changed.",
            "The toy train was going in circles around the Christmas tree.",
            "going in circles",
        ),
        (
            "They hit a wall in the negotiations and could not proceed.",
            "They hit a wall while playing ball in the narrow alleyway.",
            "hit a wall",
        ),
        (
            "Her career took off after she published her first novel.",
            "The plane took off after a short delay on the runway.",
            "took off",
        ),
        (
            "He was lost in a sea of unfamiliar faces at the conference.",
            "The fisherman was lost in a sea of fog miles from the shore.",
            "lost in a sea of",
        ),

        # --- Growth / organic mappings ---
        (
            "The idea took root in her mind and she could not stop thinking.",
            "The seedling took root in the garden after a week of watering.",
            "took root",
        ),
        (
            "Their friendship blossomed over the course of that summer.",
            "The cherry tree blossomed beautifully in the warm spring sun.",
            "blossomed",
        ),
        (
            "The company's growth was stunted by excessive regulation.",
            "The tree's growth was stunted by the lack of sunlight.",
            "growth was stunted",
        ),
        (
            "Seeds of doubt were planted in his mind by her questions.",
            "Seeds of lavender were planted in the garden by the fence.",
            "seeds of",
        ),
        (
            "His confidence withered under the constant stream of criticism.",
            "The flowers withered under the scorching August heat.",
            "withered under",
        ),

        # --- Water / flow mappings ---
        (
            "A wave of nostalgia washed over her when she heard the song.",
            "A wave of cold water washed over her feet on the beach.",
            "wave of",
        ),
        (
            "Information flowed freely between the two departments.",
            "Water flowed freely from the broken pipe in the basement.",
            "flowed freely",
        ),
        (
            "She was flooded with memories of her childhood home.",
            "The basement was flooded with water after the heavy rain.",
            "flooded with",
        ),
        (
            "His emotions poured out during the heartfelt speech.",
            "The rainwater poured out of the overflowing gutter.",
            "poured out",
        ),

        # --- Construction / structure mappings ---
        (
            "She built a strong case for the defense with solid evidence.",
            "She built a strong fence around the property with solid posts.",
            "built a strong",
        ),
        (
            "The foundation of their marriage was trust and honesty.",
            "The foundation of the building was cracked and needed repair.",
            "foundation of",
        ),
        (
            "His argument collapsed under careful examination.",
            "The old barn collapsed under the weight of the snow.",
            "collapsed under",
        ),
        (
            "They constructed an elaborate plan to surprise their parents.",
            "They constructed an elaborate treehouse in the old oak tree.",
            "constructed an elaborate",
        ),

        # --- Fabric / texture mappings ---
        (
            "The fabric of society was torn apart by the political crisis.",
            "The fabric of the old curtain was torn apart by the wind.",
            "fabric of",
        ),
        (
            "She wove a compelling narrative throughout her entire speech.",
            "She wove a colorful blanket throughout the cold winter months.",
            "wove a",
        ),
        (
            "His story was full of loose threads that did not add up.",
            "The sweater was full of loose threads after years of wear.",
            "loose threads",
        ),

        # --- Food / consumption mappings ---
        (
            "He devoured the new novel in a single afternoon sitting.",
            "He devoured the entire pizza in a single sitting at lunch.",
            "devoured",
        ),
        (
            "She had to digest the shocking news before she could respond.",
            "She had to digest the heavy meal before she could exercise.",
            "digest",
        ),
        (
            "The professor fed them a steady diet of challenging problems.",
            "The farmer fed them a steady diet of grain and fresh water.",
            "fed them a steady diet of",
        ),

        # --- Battle / conflict mappings ---
        (
            "She fought her way through a mountain of paperwork that week.",
            "She fought her way through the dense underbrush in the forest.",
            "fought her way through",
        ),
        (
            "He defended his position in the debate with strong evidence.",
            "He defended his position on the hilltop against the attackers.",
            "defended his position",
        ),
        (
            "Her words struck a nerve and he stormed out of the meeting.",
            "The baseball struck a nerve in his elbow and he dropped the bat.",
            "struck a nerve",
        ),

        # --- Vision / understanding mappings ---
        (
            "I see what you mean now that you have explained it clearly.",
            "I see the mountain now that the clouds have finally cleared.",
            "see",
        ),
        (
            "She looked at the problem from a completely different angle.",
            "She looked at the painting from a completely different angle.",
            "from a completely different angle",
        ),
        (
            "He has a deep understanding of quantum mechanics.",
            "The submarine reached a deep point in the ocean trench.",
            "deep",
        ),

        # --- Music / sound mappings ---
        (
            "Her proposal struck a chord with the entire committee.",
            "The guitarist struck a chord and the crowd started cheering.",
            "struck a chord",
        ),
        (
            "His speech resonated with voters across the entire country.",
            "The bell resonated through the valley for several seconds.",
            "resonated",
        ),
    ]

    for metaphorical_text, literal_text, phrase in _paired:
        pairs.append(MetaphorPair(
            text=metaphorical_text,
            label='metaphorical',
            pair_id=pid,
            metaphor=phrase,
        ))
        pairs.append(MetaphorPair(
            text=literal_text,
            label='literal',
            pair_id=pid,
            metaphor=phrase,
        ))
        pid += 1

    # =====================================================================
    # CONTROLS: complex, vivid, fully compositional descriptive language
    # No metaphor — just precise literal description
    # =====================================================================

    _controls = [
        "The mathematician solved the equation by substituting variables systematically.",
        "The architect measured each wall twice before marking the positions.",
        "The surgeon carefully aligned the bone fragments under the microscope.",
        "The mechanic replaced all four brake pads and tested the hydraulic system.",
        "The librarian organized the returns by category and shelf number.",
        "The pilot checked every instrument on the dashboard before taxiing.",
        "The programmer traced the memory leak to an unclosed file handle.",
        "The chemist combined the two reagents in a graduated cylinder.",
        "The electrician tested the circuit with a multimeter before reconnecting.",
        "The tailor adjusted the seam by exactly three millimeters on each side.",
        "The geologist examined the rock layers exposed by the road cut.",
        "The baker measured the flour and sugar on a digital kitchen scale.",
        "The plumber tightened the pipe fitting with a wrench until it stopped dripping.",
        "The photographer adjusted the aperture and shutter speed for the low light.",
        "The carpenter sanded the surface smooth and applied two coats of varnish.",
        "The astronomer recorded the star positions at three different times.",
        "The biologist counted the cell colonies under the stereo microscope.",
        "The locksmith filed the key blank until it matched the lock cylinder.",
        "The cartographer plotted the coordinates on the topographic map grid.",
        "The jeweler polished the setting and checked the stone with a loupe.",
        "The welder joined the two steel beams at a ninety degree angle.",
        "The optician ground the lens to the precise curvature specified.",
        "The accountant reconciled the ledger entries against the bank statement.",
        "The beekeeper inspected each frame in the hive for signs of disease.",
        "The glazier scored the glass pane and snapped it along the line.",
        "The ceramicist shaped the clay on the wheel and trimmed the base.",
        "The typesetter aligned each line of text to the margin guides.",
        "The vintner tested the acidity of the juice with a hydrometer.",
        "The watchmaker replaced the mainspring with tweezers and a loupe.",
        "The navigator plotted the course using compass bearings and tide charts.",
    ]

    for text in _controls:
        pairs.append(MetaphorPair(
            text=text, label='control', pair_id=pid, metaphor='',
        ))
        pid += 1

    return pairs


def by_label(pairs: List[MetaphorPair]) -> dict:
    """Group sentence pairs by label."""
    groups = {'metaphorical': [], 'literal': [], 'control': []}
    for p in pairs:
        groups[p.label].append(p)
    return groups


def get_paired_only(pairs: List[MetaphorPair]) -> List[MetaphorPair]:
    """Filter to only paired metaphorical/literal examples (strongest test)."""
    from collections import Counter
    id_counts = Counter(p.pair_id for p in pairs)
    paired_ids = {pid for pid, count in id_counts.items() if count >= 2}
    return [p for p in pairs if p.pair_id in paired_ids]
