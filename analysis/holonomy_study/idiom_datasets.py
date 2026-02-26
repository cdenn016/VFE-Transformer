"""
Idiom / Literal Sentence Pairs
================================

Curated dataset for testing non-compositionality as gauge curvature.

Design principles:
    - Each pair shares identical surface structure (same idiom phrase),
      but one uses the phrase idiomatically and the other literally.
    - "Kick the bucket" (idiomatic = die) vs "kick the bucket across the floor"
      (literal = physically kick a bucket).
    - Non-compositionality IS curvature: the whole meaning can't be recovered
      from transporting parts independently. Parallel transport through
      "kick" then "bucket" doesn't compose to "die".
    - Controls: complex compositional language (metaphors that ARE compositional,
      garden-path sentences, unusual-but-literal phrasing).

Labels:
    'idiomatic' — non-compositional meaning (high expected curvature)
    'literal'   — compositional meaning of the same phrase (low expected curvature)
    'control'   — complex but compositional language
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class IdiomPair:
    """A labeled sentence for idiom holonomy measurement."""
    text: str
    label: str          # 'idiomatic', 'literal', or 'control'
    pair_id: int        # links idiomatic/literal pairs
    idiom: str = ''     # the shared idiom phrase (if applicable)


def load_idiom_pairs() -> List[IdiomPair]:
    """
    Load curated idiom/literal/control dataset.

    Returns list of IdiomPair with balanced labels.
    """
    pairs = []
    pid = 0

    # =====================================================================
    # PAIRED: same phrase, idiomatic vs literal usage
    # These are the strongest test — identical words, different compositionality
    # =====================================================================

    _paired = [
        # (idiomatic_sentence, literal_sentence, idiom_phrase)
        (
            "After years of illness, the old man finally kicked the bucket.",
            "The child kicked the bucket across the yard and it rolled into the garden.",
            "kicked the bucket",
        ),
        (
            "She let the cat out of the bag about the surprise party.",
            "She let the cat out of the bag and it ran straight into the kitchen.",
            "let the cat out of the bag",
        ),
        (
            "He decided to bite the bullet and ask for a raise.",
            "The soldier had to bite the bullet while the medic removed the shrapnel.",
            "bite the bullet",
        ),
        (
            "The company decided to cut corners on the new building project.",
            "The tailor had to cut corners on the fabric to make the pattern fit.",
            "cut corners",
        ),
        (
            "When the project failed, he had to face the music from his boss.",
            "The conductor turned around to face the music and raised his baton.",
            "face the music",
        ),
        (
            "She was feeling under the weather and stayed home from work.",
            "The hikers found shelter under the weather station on the ridge.",
            "under the weather",
        ),
        (
            "The politician tried to pull the wool over our eyes with empty promises.",
            "The farmer pulled the wool over the fence to dry it in the sun.",
            "pull the wool over",
        ),
        (
            "After the argument, she gave him the cold shoulder at dinner.",
            "The butcher gave him the cold shoulder of lamb from the refrigerator.",
            "gave him the cold shoulder",
        ),
        (
            "The startup was on its last legs after losing its main investor.",
            "The old table was on its last legs and wobbled whenever you touched it.",
            "on its last legs",
        ),
        (
            "He really put his foot in his mouth at the dinner party.",
            "The baby put his foot in his mouth and started chewing on his toes.",
            "put his foot in his mouth",
        ),
        (
            "The new employee was eager to break the ice at the team meeting.",
            "The fishermen used hammers to break the ice on the frozen lake.",
            "break the ice",
        ),
        (
            "She decided to burn her bridges by telling her boss exactly what she thought.",
            "The retreating army decided to burn the bridges behind them to slow the enemy.",
            "burn bridges",
        ),
        (
            "He was barking up the wrong tree if he thought she was responsible.",
            "The dog was barking up the wrong tree while the cat hid in the bushes.",
            "barking up the wrong tree",
        ),
        (
            "The government tried to sweep the scandal under the rug before the election.",
            "She quickly swept the crumbs under the rug before the guests arrived.",
            "sweep under the rug",
        ),
        (
            "They were back to square one after the deal fell through.",
            "The children moved their pieces back to square one to restart the board game.",
            "back to square one",
        ),
        (
            "She spilled the beans about the merger during lunch.",
            "He accidentally spilled the beans all over the kitchen counter.",
            "spilled the beans",
        ),
        (
            "The project manager told the team not to rock the boat before the deadline.",
            "The children started to rock the boat and water splashed over the sides.",
            "rock the boat",
        ),
        (
            "He hit the nail on the head with his analysis of the problem.",
            "She hit the nail on the head with the hammer and drove it into the wood.",
            "hit the nail on the head",
        ),
        (
            "The detective told his partner not to jump the gun on the arrest.",
            "The sprinter was disqualified for jumping the gun at the starting line.",
            "jump the gun",
        ),
        (
            "She had to bite her tongue when her colleague took credit for her work.",
            "He accidentally bit his tongue while eating and it started to bleed.",
            "bite tongue",
        ),
        (
            "After the scandal, the CEO was left hanging out to dry by the board.",
            "She left the laundry hanging out to dry on the clothesline in the yard.",
            "hanging out to dry",
        ),
        (
            "He was skating on thin ice with his repeated violations of company policy.",
            "The children were skating on thin ice at the edge of the pond.",
            "skating on thin ice",
        ),
        (
            "She threw in the towel after months of fruitless negotiations.",
            "The boxer's trainer threw in the towel to stop the fight in the eighth round.",
            "threw in the towel",
        ),
        (
            "The team had to go back to the drawing board after the prototype failed.",
            "The architect went back to the drawing board to revise the floor plan.",
            "back to the drawing board",
        ),
        (
            "He was caught red handed stealing office supplies from the storage room.",
            "The child was caught red handed after finger painting all over the walls.",
            "caught red handed",
        ),
        (
            "She turned a blind eye to the accounting irregularities for months.",
            "He turned his blind eye toward the window, unable to see the garden anymore.",
            "turned a blind eye",
        ),
        (
            "The report buried the lead by starting with background information.",
            "The mining crew buried the lead pipes three feet below the surface.",
            "buried the lead",
        ),
        (
            "He threw her under the bus during the investigation to save himself.",
            "The luggage fell off the rack and he threw it under the bus for storage.",
            "threw under the bus",
        ),
        (
            "She decided to bury the hatchet and forgive her sister after the argument.",
            "The lumberjack decided to bury the hatchet in the stump at the end of the day.",
            "bury the hatchet",
        ),
        (
            "The new regulation added fuel to the fire of the ongoing protest.",
            "He added fuel to the fire and the flames rose higher in the fireplace.",
            "added fuel to the fire",
        ),
        (
            "The manager wanted everyone to pull their weight on the final push.",
            "The climbers had to pull their weight up the rope to reach the ledge.",
            "pull their weight",
        ),
        (
            "She bent over backwards to accommodate the demanding client.",
            "The gymnast bent over backwards and held the position for three seconds.",
            "bent over backwards",
        ),
        (
            "He was just a drop in the bucket compared to the other donors.",
            "A single drop fell in the bucket from the leaky roof above.",
            "drop in the bucket",
        ),
        (
            "The investigation opened a can of worms that nobody anticipated.",
            "The fisherman opened a can of worms and baited his hook by the river.",
            "opened a can of worms",
        ),
        (
            "She had to read between the lines to understand his true intentions.",
            "The editor had to read between the lines of text to find the typo.",
            "read between the lines",
        ),
        (
            "The company was cutting its teeth on the first major government contract.",
            "The puppy was cutting its teeth on everything it could find in the house.",
            "cutting its teeth",
        ),
        (
            "He had a chip on his shoulder after being passed over for the promotion.",
            "He brushed the chip off his shoulder where it had fallen from the ceiling.",
            "chip on his shoulder",
        ),
        (
            "The politician was sitting on the fence about the controversial bill.",
            "The cat was sitting on the fence watching the birds in the garden.",
            "sitting on the fence",
        ),
        (
            "She was walking on eggshells around her temperamental boss all week.",
            "He was carefully walking on eggshells for the science experiment demonstration.",
            "walking on eggshells",
        ),
        (
            "The teacher told the students to keep their eyes peeled for the answer.",
            "The surgeon had to keep the patient's eyes peeled open during the procedure.",
            "keep eyes peeled",
        ),
        # --- Batch 2: more common English idioms ---
        (
            "She was over the moon when she got the job offer.",
            "The satellite passed over the moon on its way to Mars.",
            "over the moon",
        ),
        (
            "He was feeling a bit green around the gills after the meeting.",
            "The fish was green around the gills and clearly not fresh.",
            "green around the gills",
        ),
        (
            "They decided to call it a day after twelve hours of negotiations.",
            "The referee had to call it a day when the rain made the field unplayable.",
            "call it a day",
        ),
        (
            "She had to eat humble pie after her prediction turned out wrong.",
            "The baker decided to eat the humble pie he had made from leftover ingredients.",
            "eat humble pie",
        ),
        (
            "The scandal was the last straw for the board of directors.",
            "The bartender placed the last straw in the glass and handed it to the customer.",
            "the last straw",
        ),
        (
            "He was trying to get a foot in the door at the marketing firm.",
            "She managed to get her foot in the door before it closed shut.",
            "foot in the door",
        ),
        (
            "The news spread like wildfire through the small town.",
            "The blaze spread like wildfire across the dry forest floor.",
            "spread like wildfire",
        ),
        (
            "She decided to take the bull by the horns and confront the issue directly.",
            "The matador tried to take the bull by the horns during the final act.",
            "take the bull by the horns",
        ),
        (
            "He knew he was just a small fish in a big pond at the new company.",
            "They released the small fish back in the big pond behind the cabin.",
            "small fish in a big pond",
        ),
        (
            "The politician was trying to save face after the embarrassing leak.",
            "The surgeon worked carefully to save the face of the accident victim.",
            "save face",
        ),
        (
            "It was raining cats and dogs during the entire football match.",
            "The shelter was so full it was practically raining cats and dogs from every room.",
            "raining cats and dogs",
        ),
        (
            "She had butterflies in her stomach before the big presentation.",
            "The doctor found butterflies in the specimen jar from the biology collection.",
            "butterflies in stomach",
        ),
        (
            "He let sleeping dogs lie rather than bring up the old disagreement.",
            "She walked quietly past the yard to let the sleeping dogs lie undisturbed.",
            "let sleeping dogs lie",
        ),
        (
            "The contract negotiation was a piece of cake for the experienced lawyer.",
            "She cut herself a piece of cake from the platter on the kitchen table.",
            "piece of cake",
        ),
        (
            "He was beating around the bush instead of answering the question directly.",
            "The gardener was beating around the bush with a stick to scare off the rabbits.",
            "beating around the bush",
        ),
        (
            "The whole project was hanging by a thread after the funding was cut.",
            "The ornament was hanging by a thread from the top branch of the tree.",
            "hanging by a thread",
        ),
        (
            "She smelled a rat when the numbers did not add up in the financial report.",
            "The dog smelled a rat hiding behind the wooden shed in the backyard.",
            "smelled a rat",
        ),
        (
            "He decided to turn over a new leaf after his health scare last year.",
            "She turned over a new leaf in the book to continue reading the next chapter.",
            "turn over a new leaf",
        ),
        (
            "The two companies decided to bury the hatchet and form a joint venture.",
            "The scout leader showed the children how to bury the hatchet safely at camp.",
            "bury the hatchet",
        ),
        (
            "She was at the end of her rope dealing with the difficult client.",
            "The climber was at the end of his rope and could not reach the next ledge.",
            "at the end of rope",
        ),
        (
            "He stirred the pot by sharing confidential details with the press.",
            "The chef stirred the pot slowly to keep the soup from burning on the bottom.",
            "stirred the pot",
        ),
        (
            "She was on cloud nine after receiving the acceptance letter from the university.",
            "The airplane flew just below cloud nine on its approach to the airport.",
            "on cloud nine",
        ),
        (
            "The lawyer told his client not to cry over spilled milk about the lost case.",
            "The toddler started to cry over the spilled milk that covered the kitchen floor.",
            "cry over spilled milk",
        ),
        (
            "He was dragging his feet on the merger because he feared losing control.",
            "The exhausted hiker was dragging his feet through the mud on the trail.",
            "dragging his feet",
        ),
        (
            "The CEO was playing with fire by ignoring the regulatory warnings.",
            "The child was playing with fire in the backyard and burned his finger.",
            "playing with fire",
        ),
        (
            "She went out on a limb to defend her colleague during the review.",
            "The cat went out on a limb to reach the bird sitting at the end of the branch.",
            "out on a limb",
        ),
        (
            "He was living on borrowed time after the diagnosis.",
            "The library book was on borrowed time and needed to be returned by Friday.",
            "on borrowed time",
        ),
        (
            "The proposal was dead in the water without the chairman's support.",
            "The boat was dead in the water after the engine failed in the middle of the lake.",
            "dead in the water",
        ),
        (
            "She jumped on the bandwagon and started posting daily videos online.",
            "The children jumped on the bandwagon as it rolled through the parade route.",
            "jumped on the bandwagon",
        ),
        (
            "He had to eat his words when the results proved him completely wrong.",
            "The parrot seemed to eat his words, repeating them back in a garbled fashion.",
            "eat his words",
        ),
    ]

    for idiomatic_text, literal_text, idiom in _paired:
        pairs.append(IdiomPair(
            text=idiomatic_text,
            label='idiomatic',
            pair_id=pid,
            idiom=idiom,
        ))
        pairs.append(IdiomPair(
            text=literal_text,
            label='literal',
            pair_id=pid,
            idiom=idiom,
        ))
        pid += 1

    # =====================================================================
    # CONTROLS: complex compositional language (not idiomatic)
    # Novel metaphors, garden-path, unusual-but-compositional
    # =====================================================================

    _controls = [
        # Novel metaphors (compositional — meaning derives from parts)
        "His anger was a volcano erupting after years of dormancy.",
        "The city was a beehive of activity during the holiday festival.",
        "Her smile was a lighthouse guiding him through the difficult conversation.",
        "The classroom was a zoo after the teacher left for five minutes.",
        "His memory was a sieve that let every important date slip through.",
        # Garden-path sentences (complex parsing, but fully compositional)
        "The horse raced past the barn fell down in the muddy field.",
        "The old man the boats that sail up the river every morning.",
        "The complex houses married and single soldiers and their families.",
        "The cotton clothing is made of grows in the southern states.",
        "The man who hunts ducks out on weekends when the weather is nice.",
        # Unusual but compositional phrasing
        "Seventeen purple elephants danced gracefully on the frozen pond at midnight.",
        "The mathematician ate the square root of the chocolate cake with gusto.",
        "A committee of dolphins drafted the proposal for underwater highways.",
        "The librarian organized the clouds by color and filed them alphabetically.",
        "The grandfather clock debated philosophy with the kitchen toaster at dawn.",
        # Technically precise compositional language
        "The dissolution of the polymer occurred at exactly three hundred degrees.",
        "The photosynthetic efficiency of the modified chloroplasts exceeded expectations.",
        "The tectonic plates shifted four centimeters along the fault line last year.",
        "The electromagnetic spectrum extends far beyond the visible light range.",
        "The gravitational lensing effect distorted the image of the distant galaxy.",
        # Factual surprising (compositional)
        "Octopuses have three hearts and their blood is blue because of copper.",
        "A group of flamingos standing together is officially called a flamboyance.",
        "Honey found in ancient Egyptian tombs was still edible after thousands of years.",
        "The shortest war in history lasted only thirty eight minutes between two nations.",
        "Venus spins so slowly that one day there is longer than one year.",
        # Emotionally charged but compositional
        "She cried for three hours after reading the letter from her late grandmother.",
        "The firefighter carried the child through the smoke filled hallway to safety.",
        "He stood alone at the grave long after everyone else had gone home.",
        "The soldier returned home after four years to find his daughter had grown up.",
        "She held his hand tightly as the doctor delivered the difficult diagnosis.",
        # Syntactically complex but compositional
        "The book that the professor who taught the class that I took recommended was excellent.",
        "The man who the woman who the child liked saw left the building.",
        "The report that the committee that the board appointed submitted was rejected.",
        "The house that the contractor who the bank financed built collapsed in the storm.",
        "The theory that the scientist who the university hired proposed was controversial.",
        # Narrative compositional
        "The chef accidentally added salt instead of sugar to the birthday cake batter.",
        "The train arrived at the empty platform just as the last passenger walked away.",
        "The letter sat unopened on the desk for three weeks before someone noticed it.",
        "The lighthouse keeper watched the storm approach from across the dark water.",
        "The old photograph fell out of the book and landed face up on the floor.",
        # Complex descriptive (compositional)
        "The ancient cathedral towered above the narrow cobblestone streets of the old quarter.",
        "The chemistry experiment produced a bright orange precipitate that settled to the bottom.",
        "The migration pattern of the monarch butterflies spans over three thousand miles.",
        "The archaeologists carefully brushed sand from the pottery shards at the excavation site.",
        "The double helix structure of DNA was first described in nineteen fifty three.",
        # Cause-effect chains (compositional but complex)
        "The heavy rainfall caused the river to overflow and flood the farmland downstream.",
        "The new trade agreement reduced tariffs and increased exports by fourteen percent.",
        "The volcanic eruption sent ash thirty kilometers into the atmosphere and disrupted flights.",
        "The interest rate hike slowed consumer spending and cooled the overheated housing market.",
        "The antibiotics eliminated the bacterial infection within seven days of treatment.",
        # Temporal sequences (compositional)
        "She graduated in June, started her new job in August, and moved to the city in September.",
        "The spacecraft launched at dawn, reached orbit by noon, and docked with the station at dusk.",
        "He planted the seeds in spring, watered them through summer, and harvested in early autumn.",
        "The ice age began slowly, peaked over thousands of years, and retreated just as gradually.",
        "The company was founded in a garage, expanded to an office, and eventually filled a campus.",
        # Counterfactual compositional
        "If the bridge had been built ten meters higher the flooding would not have caused damage.",
        "Had the pilot noticed the warning light sooner the emergency could have been avoided.",
        "The experiment would have succeeded if the temperature had been kept below zero degrees.",
        "She would have caught the train if she had left the house five minutes earlier.",
        "The treaty might have prevented the conflict if both sides had agreed to the terms.",
        # Scientific compositional
        "The Mars rover collected soil samples that contained traces of ancient microbial life.",
        "Quantum entanglement allows particles to share states across arbitrary distances instantly.",
        "The enzyme catalyzes the reaction by lowering the activation energy required for bonding.",
        "Plate tectonics explains why earthquakes occur along the boundaries of continental plates.",
        "The Doppler effect causes the pitch of a siren to change as it moves toward or away.",
        # Additional balance
        "The telescope revealed a cluster of galaxies twelve billion light years from Earth.",
        "The patient recovered fully after the transplant surgery lasted nine hours.",
        "The algorithm sorted ten million records in under three seconds on the new hardware.",
        "The glacier has retreated over two kilometers in the past decade due to warming.",
        "The orchestra performed the symphony without a single mistake during the live broadcast.",
    ]

    for text in _controls:
        pairs.append(IdiomPair(
            text=text, label='control', pair_id=pid, idiom=''
        ))
        pid += 1

    return pairs


def get_paired_only(pairs: List[IdiomPair]) -> List[IdiomPair]:
    """Filter to only paired idiomatic/literal examples (strongest test)."""
    from collections import Counter
    id_counts = Counter(p.pair_id for p in pairs)
    paired_ids = {pid for pid, count in id_counts.items() if count >= 2}
    return [p for p in pairs if p.pair_id in paired_ids]


def by_label(pairs: List[IdiomPair]) -> dict:
    """Group sentence pairs by label."""
    groups = {'idiomatic': [], 'literal': [], 'control': []}
    for p in pairs:
        if p.label in groups:
            groups[p.label].append(p)
    return groups
