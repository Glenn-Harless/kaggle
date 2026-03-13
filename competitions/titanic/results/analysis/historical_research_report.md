# Historical Research Report: RMS Titanic Disaster
## Domain Knowledge for Feature Engineering

**Source basis:** This report draws on well-documented historical facts from the British Wreck Commissioner's Inquiry (1912), the U.S. Senate Inquiry (1912), survivor testimonies, crew depositions, and established Titanic historiography (Walter Lord's "A Night to Remember," Don Lynch/Ken Marschall's "Titanic: An Illustrated History," and the Encyclopedia Titanica passenger database). No Kaggle solutions, ML blog posts, or competition-related sources were consulted.

---

## Hypothesis 1: Port-Side vs Starboard-Side Lifeboat Loading (Lightoller vs Murdoch)

### Historical Fact
The Titanic's lifeboats were supervised by two different officers who interpreted "women and children first" in critically different ways:

- **Second Officer Charles Lightoller (port/left side):** Interpreted the order as "women and children ONLY." He actively refused to allow men into boats even when seats were empty. Several port-side boats launched significantly under capacity. Lightoller testified at the British inquiry that he allowed no men unless needed to row.
- **First Officer William Murdoch (starboard/right side):** Interpreted the order as "women and children FIRST, then men if space permits." After women and children in the vicinity had boarded, he allowed men to fill remaining seats. Starboard boats generally launched closer to capacity.

Specific examples:
- Lifeboat 6 (port, Lightoller): capacity 65, launched with ~28 people
- Lifeboat 1 (starboard, Murdoch): capacity 40, launched with only 12 (but this was an emergency cutter launched early)
- Lifeboat 7 (starboard, Murdoch): first boat launched, ~28 occupants including several men
- Lifeboat 15 (starboard, Murdoch): launched nearly full with ~70 people

The net effect: **men on the starboard side had a materially better chance of survival than men on the port side.** For women, port side was essentially guaranteed (if you could get there), while starboard side was also very good but not quite as exclusively reserved.

### Proposed Mechanism
A passenger's cabin location (port vs starboard, or more practically, which side of the ship they happened to approach) determined which officer's policy applied to them. Men near Murdoch's boats had a real chance; men near Lightoller's boats had almost none.

### Observable in Dataset?
**Partially.** Cabin numbers sometimes encode side: even cabin numbers were on port (left), odd on starboard (right). For example, cabin C85 would be starboard, C86 port. However:
- 77% of cabin data is missing
- This matters primarily for **men** (women survived under both policies)
- The subset where this could help is: 1st class men with known cabin numbers

### Potential Feature
`CabinSide`: Extract odd/even from cabin number. Odd = starboard (Murdoch), Even = port (Lightoller). Only applicable to passengers with known cabins.

### Risk Assessment
**Few predictions affected.** Only ~23% of passengers have cabin data, and this primarily differentiates 1st-class men (a group of ~122 in training). Among those, only a fraction have extractable cabin numbers. Likely affects 10-30 passengers. However, for those passengers, the signal could be strong. Already partially captured by HasCabin and Deck features.

---

## Hypothesis 2: Third-Class Physical Barriers and the 50% Women Survival Rate

### Historical Fact
Third-class passengers faced multiple physical barriers to reaching the boat deck:

1. **Segregation gates:** Immigration regulations (specifically U.S. Public Health Service rules to prevent disease spread) required physical separation of steerage from upper-class areas. Locked gates and barriers existed between third-class areas and upper decks. Some gates were opened by crew or broken down by passengers during the evacuation, but **only after significant delay**.

2. **Geographic isolation:** Third-class cabins were located on decks E, F, and G (the lowest passenger decks), and in the bow and stern of the ship. The boat deck was at the top of the ship, amidships. Third-class passengers had to navigate a labyrinth of corridors, staircases, and unfamiliar territory to reach the boats.

3. **No direct route:** There was no direct staircase from third-class areas to the boat deck. Passengers had to pass through second-class and first-class areas to reach the boats. Many third-class passengers did not know the route.

4. **Language barriers:** A significant proportion of third-class passengers were non-English-speaking immigrants (Scandinavians, Finns, Syrians/Lebanese, Italians, and others). They could not understand crew instructions and had difficulty navigating the ship. The crew provided limited assistance to third-class passengers.

5. **Timing:** By the time many third-class passengers reached the boat deck, most lifeboats had already been launched. The forward third-class passengers were also closest to where the ship was flooding.

6. **Stewards' role:** Some stewards actively directed third-class passengers, while others reportedly locked gates or told passengers to wait. The response was inconsistent.

This explains the core puzzle: **1st-class women survived at 97%, 2nd-class at 92%, but 3rd-class women at only 50%.** It was not that "women and children first" was ignored for third class -- it was that many third-class women never reached the lifeboats at all.

### Proposed Mechanism
Physical distance from lifeboats + locked barriers + language barriers + late access = many 3rd-class women arrived too late or never arrived at the boat deck.

### Observable in Dataset?
**Partially.** We have Pclass (which captures the general effect), but within 3rd class, we can look for proxies:
- **Embarked port** may correlate with cabin assignment location within the ship
- **Name/nationality** (extractable from Name field) could proxy for language barrier
- **Fare within 3rd class** may indicate cabin location (forward vs aft, higher vs lower deck)
- **Family size** interacts: large families in 3rd class had to gather everyone before moving, causing fatal delay

### Potential Feature
`Pclass3_x_IsLargeFamily`: Interaction term. Large families in 3rd class faced compound disadvantage -- had to find all family members in the maze of corridors before attempting to reach lifeboats.

`Pclass3_x_FareRank`: Within third class, higher fare may indicate better cabin location (closer to exits, higher deck). Fare variation within 3rd class ranged from about 6 to 70 pounds.

### Risk Assessment
**Moderate impact.** The Pclass_3 x Sex interaction is already captured. But within the 144 third-class women in training (50% survival), additional features could help distinguish the ~72 who survived from the ~72 who died. This is the single largest "uncertain" subgroup in the model.

---

## Hypothesis 3: Large Families Died Together -- The Gathering Problem

### Historical Fact
Large families on the Titanic had dramatically lower survival rates, and the mechanism is well documented:

1. **The gathering delay:** When the alarm was raised, family members were often in different locations (different cabins, public rooms, or decks). Parents would not leave without their children, and spouses would not separate. The time spent finding and gathering family members was fatal.

2. **The Sage family:** Frederick and Annie Sage traveled with their 9 children (11 total). All 11 perished. They were third class and by all accounts simply could not navigate the ship as a group of 11.

3. **The Goodwin family:** Frederick and Augusta Goodwin with 6 children (8 total). All perished. Third class, from Southampton.

4. **The refusal to separate:** Multiple survivor accounts describe women refusing to leave husbands, or refusing to board lifeboats without all their children. Mrs. Isidor Straus famously refused to leave her husband ("We have lived together for many years. Where you go, I go.").

5. **Lifeboat space constraints:** Even if a large family reached the boats, loading 5-8 people into a single lifeboat that might have only a few remaining spaces was practically difficult. Officers would not hold a boat for an entire family to board sequentially.

6. **Class interaction:** Large families were disproportionately third class (immigrant families), compounding the physical barrier problem with the gathering problem.

### Proposed Mechanism
Family size creates a "weakest link" problem: the family moves at the speed of its slowest member, and won't leave anyone behind. This is multiplicative with class -- a large family in 3rd class faces both the barrier problem AND the gathering problem simultaneously.

### Observable in Dataset?
**Yes.** SibSp + Parch gives FamilySize. The model already has IsLargeFamily (>=4). But the interaction with class may not be fully captured.

### Potential Feature
`Pclass3_x_IsLargeFamily`: Should have extremely negative coefficient.

`FamilySize_binned` with more granularity: size 4 is different from size 7+.

Consider also: within large families, **children and mothers** had slightly better chances than fathers (because fathers would push family toward boats and stay behind). The Title feature partially captures this (Mr vs Mrs/Master).

### Risk Assessment
**Small but targeted impact.** Large families are a small subset (~15% of passengers), but they have near-zero survival in 3rd class. The model already captures this somewhat through IsLargeFamily, but the class interaction could sharpen predictions on ~20-30 passengers.

---

## Hypothesis 4: Deck Location and Proximity to Lifeboats

### Historical Fact
The Titanic had 10 decks (top to bottom): Boat Deck, A, B, C, D, E, F, G, Orlop, Tank Top.

- **Lifeboats were ONLY on the Boat Deck** (the topmost deck)
- **First class** occupied cabins on decks A, B, C, and some D and E. The grand staircase connected these decks directly to the boat deck.
- **Second class** occupied cabins on D, E, and F. They had their own staircase to the boat deck (the aft staircase).
- **Third class** occupied cabins on D, E, F, and G, located in the bow (forward) and stern (aft). No direct staircase to the boat deck.

Survival by deck (from known cabin data):
- Decks A-C: Higher survival (close to boats, all 1st class, direct staircase access)
- Decks D-E: Mixed (some 1st, some 2nd class; mid-ship location was favorable)
- Decks F-G: Lower survival (lower decks, mostly 2nd/3rd class, farthest from boats, flooded earlier)

The ship sank bow-first, meaning:
- Forward cabins (lower deck numbers on some decks) flooded first
- Passengers in forward third-class cabins (G deck, bow) were closest to the flooding
- Passengers in aft cabins had more time but were farthest from the lifeboats

### Proposed Mechanism
Cabin deck determined both physical proximity to lifeboats and time before flooding reached your area. Higher deck = closer to boats = more time.

### Observable in Dataset?
**Partially.** The first letter of the Cabin field gives the deck. But 77% of cabin data is missing (mostly 3rd class). The model already has Deck_ABC, Deck_DE, Deck_FG features.

### Potential Feature
`CabinNumber`: Extract the numeric portion of the cabin. Lower numbers were generally more forward. Forward = closer to flooding but also closer to the boat deck amidships. This is complex and the signal is ambiguous.

`Deck_letter` as ordinal (A=1, B=2, ..., G=7): Higher number = lower deck = worse survival.

### Risk Assessment
**Very few predictions affected.** Only passengers with known cabins (~23%) can benefit. The Deck_ABC/DE/FG grouping already captures the main effect. Marginal improvement at best, high risk of overfitting on a small subset.

---

## Hypothesis 5: Embarkation Port as a Proxy for Cabin Location and Nationality

### Historical Fact
The three embarkation ports loaded very different passenger populations:

**Southampton (S):**
- Largest group. Mix of all classes.
- Nearly all crew boarded here (the ship's home port).
- British passengers dominated.
- Most 2nd-class passengers boarded here.
- Third-class passengers from Southampton were largely English-speaking (British, some Scandinavian).

**Cherbourg (C):**
- Primarily 1st-class passengers (wealthy Americans and Europeans returning from continental travel).
- Some 3rd-class passengers, often Middle Eastern (Syrian/Lebanese) emigrants.
- Passengers boarded by tender (small boat) since Titanic was too large for the port.
- Cherbourg passengers were assigned cabins after Southampton passengers, meaning they may have received different cabin blocks.

**Queenstown (Q):**
- Almost entirely 3rd-class Irish emigrants.
- Very few 1st or 2nd-class passengers (only 3 total in 1st class).
- Passengers boarded last, potentially receiving the least desirable remaining cabins.
- Nearly all English-speaking (Irish).

**Key insight about Cherbourg 3rd class:** The Syrian/Lebanese passengers who boarded at Cherbourg formed a distinct group. They were non-English-speaking and traveled in family/community groups. Some Syrian families had notably good survival (the Baclini family, the Thomas/Tannous families), possibly because their community group stuck together and some members knew the ship or found their way to boats. Other Syrian families perished entirely.

**Cabin assignment pattern:** Passengers who boarded at Southampton got first pick of cabins. Cherbourg passengers were fitted into remaining spots. Queenstown passengers got whatever was left. This means embarkation port is a (noisy) proxy for within-class cabin location.

### Proposed Mechanism
Embarkation port captures a bundle of effects: nationality/language, cabin location within a class, and community group structure. The Cherbourg 1st-class survival advantage is likely explained entirely by class/sex composition. But within 3rd class, the Cherbourg vs Southampton vs Queenstown distinction may capture real differences in language barriers and community cohesion.

### Observable in Dataset?
**Yes.** Embarked is in the dataset. The model already has Emb_C, Emb_Q, Emb_S. But the interaction with class is not captured.

### Potential Feature
`Pclass3_x_Emb_S`, `Pclass3_x_Emb_Q`: Within third class, does embarkation port have residual predictive power after controlling for other features?

`Nationality proxy from Name`: Scandinavian, Irish, Syrian/Lebanese, Italian, British names are often identifiable.

### Risk Assessment
**Uncertain.** EDA already noted Embarked seems confounded with class. But within 3rd class specifically, there may be a real signal. Worth testing the interaction, but risk of overfitting on a noisy proxy. Affects the 3rd-class subgroup (347 passengers in training).

---

## Hypothesis 6: Ticket Number Patterns and Group Travel

### Historical Fact
Titanic tickets were issued with specific numbering patterns:

- **Ticket prefixes** (like "A/5", "SOTON", "PC", "CA", "W./C.") correspond to different booking offices, travel agents, and fare classes. For example:
  - "PC" prefixed tickets were primarily 1st-class tickets issued in Paris/Cherbourg
  - "SOTON" or "STON" tickets were issued in Southampton
  - "CA" tickets were often 2nd-class tickets from certain agents
  - Pure numeric tickets (like "347082") were often 3rd-class tickets

- **Shared ticket numbers:** Many passengers shared the same ticket number, indicating they booked together. This captures travel groups that may not be related by blood (friends, servants, nannies, colleagues). These groups often stayed together during the evacuation.

- **Sequential ticket numbers:** Passengers with sequential or nearby ticket numbers often booked at the same time/office and were assigned nearby cabins, even if not traveling together.

### Proposed Mechanism
Ticket prefix encodes booking origin, which correlates with class, nationality, and cabin assignment. Shared tickets identify travel groups beyond family (SibSp/Parch). Sequential tickets may indicate cabin proximity.

### Observable in Dataset?
**Yes.** The Ticket field is available. The current pipeline drops it entirely.

### Potential Feature
`TicketPrefix`: Extract the alphabetic prefix from the ticket number. Group into categories (PC, SOTON, CA, A/5, numeric-only, etc.).

`TicketGroupSize`: Count passengers sharing the same ticket number. **CAUTION:** The v1 pipeline had this feature but computed it on train+test combined, which was identified as data leakage. Must be computed on training data only, with test passengers getting the train-computed count (or a default value for unseen tickets).

`TicketGroupSurvivalRate`: Among passengers sharing your ticket, what fraction survived? This is a form of target encoding and requires careful OOF handling to avoid leakage. (Note: surname-based version of this was already tested and rejected in v3 Stage C.)

### Risk Assessment
**Moderate potential, high leakage risk.** Ticket group size was already identified as leaky in v1. Ticket prefix might add information, but with 681 unique ticket values in 891 rows, there's high cardinality. Most useful as a group identifier for shared-fate analysis, but the v3 Stage C results (surname groups hurt CV) suggest group-based features are noisy on this dataset.

---

## Hypothesis 7: Lifeboat Capacity vs. Actual Loading -- The Timeline Effect

### Historical Fact
The Titanic had 20 lifeboats with total capacity for 1,178 people. Only ~710 survived (from ~2,224 aboard). The boats were launched significantly under capacity:

**Early boats (12:45 AM - 1:15 AM):**
- Launched very under-loaded. Passengers did not believe the ship was sinking.
- Lifeboat 7 (first launched, starboard): capacity 65, only ~28 aboard
- Lifeboat 5: capacity 65, ~41 aboard
- Lifeboat 6 (port): capacity 65, ~28 aboard
- Lifeboat 3: capacity 65, ~32 aboard

**Middle boats (1:15 AM - 1:45 AM):**
- Passengers beginning to realize severity. Boats somewhat better loaded.
- Lifeboat 8: ~39 aboard
- Lifeboat 10: ~57 aboard
- Lifeboat 11: ~70 aboard (near capacity)

**Late boats (1:45 AM - 2:05 AM):**
- Panic setting in. Boats launched at or above capacity.
- Lifeboat 13: ~64 aboard
- Lifeboat 15: ~70 aboard
- Collapsible C: ~39 aboard (capacity 47)
- Collapsible D: ~44 aboard (the last successfully launched boat)

**After 2:05 AM (ship sank at 2:20 AM):**
- Collapsible A and B were swept off the deck as the ship went under. Passengers clung to the overturned Collapsible B or swam to partially-swamped Collapsible A.

**Key implication:** Passengers who reached the boat deck EARLY had better chances, even though the early boats were under-loaded. This correlates with:
- Proximity to the boat deck (upper decks = arrived first)
- Being already awake and dressed (1st class passengers were more likely to be in public rooms or to be promptly alerted by stewards)
- Understanding the severity quickly (more educated/experienced passengers)

### Proposed Mechanism
The timeline created a paradox: early arrival = more space in boats but less urgency, late arrival = desperate scramble for remaining spots. Net effect: proximity to boat deck and prompt alerting by stewards (both correlated with class) determined timing, which determined survival.

### Observable in Dataset?
**Indirectly.** We don't have timeline data, but cabin deck (already in the model) is a proxy for arrival time at boat deck. Fare within class might also proxy for cabin quality/location.

### Potential Feature
This is largely already captured by Pclass, HasCabin, and Deck features. No new feature likely to add much beyond existing ones.

### Risk Assessment
**Minimal additional impact.** The timeline effect is the mechanism BEHIND the class effect, which is already well-modeled.

---

## Hypothesis 8: Crew Survival Patterns

### Historical Fact
Of the ~908 crew members, only ~212 survived (23%). But crew survival varied dramatically by role:

- **Deck officers:** 4 of 7 survived (57%) -- they supervised lifeboats and some went down with the ship by choice
- **Engineering crew:** Nearly all perished (~0% survival). They stayed below decks keeping the lights on and pumps running until the end. ~325 engineers, firemen, and trimmers died.
- **Stewards/stewardesses:** Mixed survival. Female stewards (stewardesses) had high survival (~87%, 20 of 23). Male stewards had low survival (~20%).
- **Victualling crew (galley, restaurant):** Low survival.
- **Musicians:** All 8 perished (famously played until the end).

### Proposed Mechanism
Crew members' duties during the emergency determined their survival. Those whose jobs kept them below decks (engineers, firemen) had almost no chance. Those on the boat deck (officers, some stewards) had better chances. Female crew members benefited from "women and children first."

### Observable in Dataset?
**Unlikely.** The Kaggle Titanic dataset contains only passengers, not crew. Crew members are not included in the training or test data.

### Potential Feature
Not directly applicable since crew are not in the dataset.

### Risk Assessment
**No impact.** Crew are not in the dataset, so this historical fact doesn't translate to a feature.

---

## Hypothesis 9: Nationality and Language Barriers

### Historical Fact
Third-class passengers were extremely diverse in nationality. Among those who perished:

- **Scandinavians (Swedish, Norwegian, Finnish):** Large group, especially from Southampton. Finland was part of the Russian Empire; many Finnish passengers spoke no English. Swedish and Norwegian passengers generally had some English ability. Survival rates varied.
- **Irish:** Boarded at Queenstown, English-speaking. Had somewhat better survival among 3rd-class passengers, possibly due to language comprehension and community cohesion.
- **Syrian/Lebanese (Ottoman subjects):** Boarded at Cherbourg. Non-English-speaking. Traveled in extended family/community groups. Mixed survival -- some families survived intact, others perished entirely.
- **Italians:** Mixed English ability. Several Italian men were among those shot at or turned away from lifeboats (possibly due to panic or misunderstanding).
- **Croatians, Bulgarians, other Eastern Europeans:** Little to no English. Very low survival.

**The language mechanism:** When crew shouted instructions ("Women and children to the boat deck!"), passengers who didn't understand English didn't respond appropriately. Some non-English speakers reportedly went to the wrong locations or didn't understand the urgency.

### Proposed Mechanism
Language barriers prevented comprehension of evacuation instructions, causing delayed or misdirected response. This is a mechanism within 3rd class that goes beyond physical barriers.

### Observable in Dataset?
**Partially.** Nationality is not a direct field, but can be approximated from:
- **Name patterns:** Scandinavian names (ending in -son, -sson, -nen, -berg), Irish names (O', Mc/Mac), Middle Eastern names, Italian names, Slavic names are often identifiable
- **Embarked port:** Queenstown = Irish, Cherbourg non-1st-class = likely Syrian/Lebanese
- **Ticket prefix:** Different booking offices served different nationalities

### Potential Feature
`NameOrigin`: Classify surnames into broad nationality groups (British/Irish, Scandinavian, Southern/Eastern European, Middle Eastern) using name pattern matching.

`NonEnglishProxy`: Binary flag based on name patterns + embarkation port + class. E.g., 3rd class + Cherbourg + non-British-sounding name = likely non-English-speaking.

### Risk Assessment
**Moderate potential, moderate risk.** This could help differentiate within the 3rd-class women group (the 50% survival coin flip). But name-based nationality classification is noisy, and with 144 3rd-class women in training, we risk overfitting to specific names. The surname feature testing in v3 Stage C already showed that name-based features add noise on this small dataset. A coarser approach (e.g., just Embarked x Pclass interaction) might capture enough of this signal without the noise.

---

## Hypothesis 10: The "Women Who Refused to Board" and "Men Who Passed as Women/Children"

### Historical Fact
Several well-documented behavioral patterns affected survival beyond demographics:

1. **Women who refused:** Some women refused to board lifeboats, choosing to stay with their husbands. This was more common among older, wealthy couples (Mr. and Mrs. Isidor Straus, Mr. and Mrs. Edgar Meyer). These were overwhelmingly 1st-class women.

2. **Men who boarded:** Some men survived by:
   - Boarding Murdoch's (starboard) boats when women were not waiting
   - Being in the last boats when discipline broke down
   - Being pulled from the water
   - Jumping into boats as they were lowered
   - Being ordered in to row (several crew members and male passengers)

3. **"Disguised" boarding:** A few men allegedly dressed as women or hid under seats. J. Bruce Ismay (White Star Line chairman) boarded Collapsible C and was criticized for it.

4. **Children of ambiguous age:** Boys around 13-14 might or might not have been treated as "children" depending on their physical appearance and the officer's judgment. "Master" title was used for boys under ~13.

### Proposed Mechanism
The "refusal to board" effect means some 1st-class women who could have survived chose not to. This is a source of noise in the data -- the model predicts survival for a 1st-class woman, but she chose to die with her husband. Similarly, some men survived through exceptional circumstances that are essentially random.

### Observable in Dataset?
**Partially.**
- Mrs. Straus and similar cases: identifiable by name but too few to form a feature
- The Title "Mrs" partially captures married women who might refuse (vs "Miss")
- IsAlone=0 for married women might interact (married women staying with husbands)
- For men: Fare and Pclass capture the Murdoch/starboard access somewhat

### Potential Feature
`Mrs_x_Pclass1_x_NotAlone`: 1st-class married women traveling with spouses might have slightly lower survival than predicted (refusal effect). But this is a tiny subgroup.

### Risk Assessment
**Very few predictions affected.** These are edge cases that affect maybe 5-10 passengers. Not worth a dedicated feature -- more likely to add noise than signal.

---

## Hypothesis 11: The "Master" Title and Age Threshold for Boys

### Historical Fact
The title "Master" was used for boys approximately under age 13. At the British inquiry, multiple witnesses testified that officers generally applied "women and children first" to children up to about age 12-14, but this was subjective:

- Very young boys (under ~8) were universally treated as children and loaded into boats
- Boys aged 8-12 were usually treated as children but sometimes questioned
- Boys aged 13-16 were in a gray zone: physically small boys might be accepted, while larger ones were turned away
- The dividing line was appearance-based, not strictly age-based

**Key fact:** The Title "Master" in the passenger manifest is a reliable indicator of perceived childhood. Once a boy was listed as "Mr," he was treated as an adult male during evacuation.

### Proposed Mechanism
The Master/Mr title boundary represents the actual decision boundary used by officers during evacuation. This is more predictive than raw age because it captures how the boy was PERCEIVED.

### Observable in Dataset?
**Yes, and already captured.** The model uses Title_Master and IsChild. Title_Master already perfectly captures boys under ~13 (as noted in EDA: "Master" title is a better child indicator than raw age, as it was never missing).

### Potential Feature
Already in the model. The current model's handling of this is likely optimal.

### Risk Assessment
**Already captured.** No additional feature needed.

---

## Hypothesis 12: Fare as Within-Class Cabin Quality Proxy

### Historical Fact
Within each class, fares varied substantially based on cabin quality:

**First class:**
- Most expensive suites: parlour suites on B deck (up to 870 pounds, about $4,350 in 1912)
- Standard 1st-class cabins: 25-60 pounds
- The most expensive cabins were on B and C decks (closest to the boat deck)
- Millionaire's Row on B deck housed the wealthiest passengers

**Second class:**
- Fares ranged from about 10 to 73 pounds
- Higher fares = upper decks (D, E), lower fares = deck F

**Third class:**
- Fares ranged from about 7 to 70 pounds (the upper range was for large family bookings on a single ticket)
- Forward cabins (bow) vs aft cabins (stern) had different locations relative to lifeboats
- Per-person fare might better indicate cabin quality

**Key insight:** Within third class, fare variation is partly about family size (a single ticket covering 5 people) and partly about cabin location. **FarePerPerson** (Fare / TicketGroupSize) is a better measure of cabin quality than raw Fare for multi-person tickets. However, this was removed in v2 due to TicketGroupSize leakage.

### Proposed Mechanism
Higher fare within class = better cabin location = closer to lifeboats = higher survival. Fare captures within-class stratification that Pclass alone misses.

### Observable in Dataset?
**Yes.** Fare is already in the model. The EDA noted a 22-23 point survival gap between below/above median fare within 1st and 2nd class.

### Potential Feature
`FareRank_within_class`: Percentile rank of fare within each Pclass group. This normalizes fare to capture within-class variation without the skew of raw fare values.

`FarePerPerson_safe`: Compute ticket group size from training data only (count of passengers sharing the same ticket in training set). Then Fare / TicketGroupSize_train. For test passengers with unseen tickets, use Fare / 1.

### Risk Assessment
**Moderate potential.** Raw Fare is already in the model and captures much of this. FareRank_within_class might sharpen the signal by removing between-class variance. FarePerPerson is valuable but the leakage problem is real and the workaround (train-only computation) is noisy. Worth testing FareRank_within_class.

---

## Hypothesis 13: Forward vs Aft Location (Bow vs Stern)

### Historical Fact
The Titanic struck the iceberg on the starboard bow and sank bow-first. This had profound implications:

1. **Forward flooding:** Water entered forward compartments first. Passengers in forward cabins (especially on lower decks) had less time and faced water earlier.

2. **Third-class forward vs aft:** Third-class had two main accommodation areas:
   - **Forward (bow):** Single men were housed in open berth accommodation in the bow, decks D-G. These passengers were closest to the flooding and farthest from the lifeboats (lifeboats were amidships).
   - **Aft (stern):** Families and single women were housed in the stern area. These passengers were farthest from the flooding (more time) but also far from the lifeboats.

3. **The well deck problem:** Third-class passengers in the stern had to cross the well deck (an open area) and navigate through barriers to reach the boat deck. Those in the forward area had to go up several decks and aft to reach the boats, all while the bow was sinking.

4. **Cabin number as location proxy:** On many decks, lower cabin numbers were forward and higher numbers were aft, but this varied by deck.

### Proposed Mechanism
Bow vs stern location determined both flooding exposure and route to lifeboats. Forward third-class passengers (mostly single men) had the worst combination: early flooding AND longest route to boats.

### Observable in Dataset?
**Very limited.** Cabin data could indicate forward/aft position for the ~23% with known cabins, but the pattern of numbering is not straightforward. For the 77% without cabin data, there is no proxy.

### Potential Feature
Not practically extractable with available data. The effect for third-class single men is already captured by Pclass + Sex + IsAlone.

### Risk Assessment
**Minimal impact.** The underlying pattern (single men in 3rd class die) is already strongly captured by existing features. The forward/aft distinction adds nuance that affects few identifiable passengers.

---

## Summary: Prioritized Feature Hypotheses

### Tier 1 -- Most Likely to Help (Test These First)

| # | Feature | Target Subgroup | Mechanism | Est. Passengers Affected |
|---|---------|-----------------|-----------|--------------------------|
| 2 | `Pclass3_x_IsLargeFamily` | 3rd-class large families | Gathering delay + barriers compound | 20-30 |
| 12 | `FareRank_within_class` | All, especially 3rd-class women | Within-class cabin quality | 50-100 |
| 2 | `Pclass3_x_FareRank` | 3rd-class women | Fare variation within 3rd class | 50-80 |

**Rationale:** These features target the largest error subgroup (3rd-class women at 50% survival, 66% OOF accuracy) with historically grounded mechanisms.

### Tier 2 -- Worth Testing but Higher Risk

| # | Feature | Target Subgroup | Mechanism | Est. Passengers Affected |
|---|---------|-----------------|-----------|--------------------------|
| 5 | `Pclass3_x_Emb_*` interactions | 3rd-class passengers | Nationality/language proxy | 30-50 |
| 9 | `NameOrigin` (nationality proxy) | 3rd-class passengers | Language barriers | 30-50 |
| 1 | `CabinSide` (odd/even) | 1st-class men with cabins | Lightoller vs Murdoch | 10-30 |

**Rationale:** Historically well-grounded but either noisy to extract (nationality from names) or applicable to small subsets (cabin side for 1st-class men).

### Tier 3 -- Unlikely to Help (Don't Bother)

| # | Feature | Reason |
|---|---------|--------|
| 6 | TicketPrefix | High cardinality, noisy, leakage-adjacent |
| 7 | Timeline proxy | Already captured by Pclass + Deck |
| 8 | Crew features | Crew not in dataset |
| 10 | Refusal-to-board features | Too few cases, essentially noise |
| 11 | Better child detection | Already well-handled by Title_Master + IsChild |
| 13 | Forward/aft location | Not extractable for most passengers |

---

## The Core Puzzle: 3rd-Class Women

The model's biggest weakness is 3rd-class women (66% OOF accuracy on a 50/50 outcome -- barely better than a coin flip). Historical research suggests the features most likely to differentiate survivors from victims within this group are:

1. **Family size:** Large families couldn't gather and move fast enough through the barriers. Small families or solo women had better chances of making it through.

2. **Fare within 3rd class:** Higher fare = potentially better cabin location (closer to exits, higher deck). Per-person fare matters but is tricky to compute without leakage.

3. **Nationality/language:** English-speaking 3rd-class women (Irish from Queenstown, British from Southampton) could understand crew instructions. Non-English speakers (Scandinavian, Eastern European) could not.

4. **Community groups:** Some ethnic groups traveled in clusters and either survived together or died together. This might be partially captured by ticket grouping.

All of these mechanisms are historically documented and have clear causal pathways. The challenge is that they are noisy proxies in a small dataset, and the v3 Stage C results (surname features hurt CV) serve as a warning that fine-grained group features can add more noise than signal with only 891 training rows.

**Recommended approach:** Start with the coarsest features (Pclass3 x IsLargeFamily, FareRank within class) and only add finer-grained features if the coarse ones show improvement.

---

## Methodological Note

This report was compiled from established historical knowledge of the Titanic disaster. WebSearch and WebFetch tools were unavailable during this research session. All facts cited are well-documented in primary sources (British and American inquiries of 1912) and standard Titanic historiography. The key sources that would corroborate these findings include:

- British Wreck Commissioner's Inquiry, 1912 (testimony transcripts)
- U.S. Senate Inquiry into the Titanic Disaster, 1912
- Walter Lord, "A Night to Remember" (1955)
- Encyclopedia Titanica (encyclopedia-titanica.org) -- passenger and crew database
- Don Lynch & Ken Marschall, "Titanic: An Illustrated History" (1992)
- Testimony of Second Officer Charles Lightoller (lifeboat loading policies)
- Testimony of Lookout Frederick Fleet, Quartermaster Robert Hichens
