# What changed when direction tokens were added?

Post-hoc saved-case synthesis · Codex — Spectral Optimizer Investigation · 11 September 2026

The strongest useful interpretation is **a compact alternative description of a population contrast can redirect a comparison beyond the literal fit examples—but it is not a reliable instruction for ordering arbitrary natural-prefix states**. Across all 64 axis/pair cases, there are 5 repeat gains, 6 repeat harms, 1 opposed/mixed case, 7 one-cohort-only changes and 45 unchanged cases. These repeated changes support taking the descriptive intervention seriously; they do not establish that the added tokens caused a particular reader's reasoning, or that their net contribution is beneficial.

The frozen overall result remains EA 74/128 versus examples-only E 75/128, cohort differences 0 and −1. There is no overall advantage here and no equivalence conclusion. This note changes neither correctness nor thresholds and introduces no semantic labels as independent evidence. “Repeat” means the same credit-difference sign in the two existing cohorts, not an independent replication study.

## Exact accounting of all 64 cells

For each axis/pair, Δ=(EA−E credit in cohort 1, EA−E credit in cohort 2). G = (+1,+1), repeat gain; H = (−1,−1), repeat harm; M = opposite nonzero signs; O = exactly one nonzero cohort difference; U = (0,0). The four bits after the code are correctness in order **EA1,E1,EA2,E2**; 1 correct, 0 incorrect. There are no exact-score ties in this frozen panel. Thus U means no within-cohort arm credit change, not necessarily agreement across cohorts. No credit classification was based on an interpretation of the texts.

| Pair | PC1 | PC2 | PC3 | PC4 |
| --- | --- | --- | --- | --- |
| N01 | U 1111 | U 0000 | U 1111 | G 1010 |
| N02 | H 0101 | U 0000 | U 0000 | U 1111 |
| N03 | U 1111 | U 1111 | H 0101 | U 1111 |
| N04 | U 1111 | U 1111 | U 1111 | H 0101 |
| N05 | U 0000 | U 0000 | O 1011 | H 0101 |
| N06 | U 1111 | O 0100 | U 1111 | U 1111 |
| N07 | U 1111 | U 0000 | U 0000 | U 0000 |
| N08 | U 1111 | U 0000 | U 1111 | G 1010 |
| N09 | M 1001 | G 1010 | O 1011 | U 1111 |
| N10 | U 0000 | U 1111 | U 1111 | U 1111 |
| N11 | U 1111 | O 0001 | U 0000 | U 1111 |
| N12 | U 1111 | U 1111 | H 0101 | O 0100 |
| N13 | G 1010 | U 0000 | U 0000 | O 0010 |
| N14 | G 1010 | U 1100 | O 1110 | U 0000 |
| N15 | U 0000 | U 1111 | U 1111 | H 0101 |
| N16 | U 0000 | U 1111 | U 1111 | U 0011 |

This exhausts 64 unique cells and 256 frozen choices. G/H contribute 10 gains/12 harms; M contributes one of each; O contributes four gains/three harms: **15 gains,16 harms,97 unchanged credits** across the 128 matched cohort cells. Of the 45 U cells, 27 have all four correct, 16 all four wrong, and two have both arms correct in one cohort and wrong in the other. All classes and both signs are retained.

## Every repeat gain and harm

L/R always denote the original dataset sides, not randomized FIRST/SECOND. Each change below occurred in both cohorts. The complete unedited prefixes appear in the final roster; the complete original A token lists and C examples are reproduced below that table. The last column is a retrospective interpretive possibility, **not observed reader reasoning or a demonstrated cause**.

| Cell | Outcome; E → EA | Retrospective interpretation and boundary |
| --- | --- | --- |
| PC1 N13 | Gain; L → R, truth R | Hand coding edits a document's representation, versus the Global Offset Table. The photographic/imaging list could make representation salient, but neither prefix explicitly describes imaging. This is a weak semantic explanation, not a clean demonstration. |
| PC1 N14 | Gain; L → R, truth R | Doneness explicitly describes a gauge; pellet grills describe cookers. An observation/measurement reading of the positive list could contrast with preparation. The tokens themselves are predominantly photographic, so that broader abstraction remains inferred. |
| PC2 N09 | Gain; R → L, truth L | Mise en place is closer to the ingredient/preparation negative description than a pizza oven. The choice is relative: the oven need not express the positive football/award tokens. This is the clearest cooking-specific plausible correction, not a domain-wide effect. |
| PC4 N01 | Gain; L → R, truth R | Dark constellations versus an astronomer's biography could favor the positive UFO/camera/apparition-associated list. That is a plausible visual-association reading, not evidence of what the reader used. |
| PC4 N08 | Gain; R → L, truth L | The dangling-else programming problem versus performance portability is compatible with the positive list's exact token “ glitches”. This is a concrete way a direction description might supply an association absent from goalkeeper/recipe examples; it does not establish a general bug detector. |
| PC1 N02 | Harm; L → R, truth L | Radio Galaxy Zoo may seem especially compatible with spacecraft/observatory/telescope associations, yet the synodic-day prefix has the higher measured score. A semantically plausible association can select the wrong endpoint ordering. |
| PC3 N03 | Harm; L → R, truth L | Modeling/optimization in JuMP could attract an evaluation/testing interpretation relative to a loop-switch antipattern. That plausible abstraction reverses the correct frozen order. |
| PC3 N12 | Harm; L → R, truth L | Rosalind's problem-solving/learning description may attract the evaluation list relative to a directive/pragma. The directive prefix nevertheless has the higher coordinate. |
| PC4 N04 | Harm; L → R, truth L | Floyd's triangle versus a voluntary initiative offering opportunities is compatible with treating provision/help as negative. The measured order instead favors Code Club. The positive list has no obvious direct triangle-specific explanation. |
| PC4 N05 | Harm; R → L, truth R | Psychology research versus a coding-class platform resembles investigation versus provision, but the platform prefix is higher. This directly limits an unrestricted observation/provision gloss at this endpoint. |
| PC4 N15 | Harm; R → L, truth R | A system-time description versus a nonprofit programming-club network again admits a descriptive/measurement versus provision reading; the network prefix is actually higher. Neither this nor N05 establishes a causal “anchoring” error. |

The one opposed case is PC1 N09: cohort 1 improves while cohort 2 worsens, with the same underlying pizza-oven/mise-en-place alternatives. The seven O cells and their full correctness patterns are visible above; none is promoted into a repeat effect. Cooking's aggregate positive uses just N09 and N14 across axes/readers and cannot rescue the primary result. The same proposed semantic gloss must face the repeat harms, not only the agreeable examples.

## All unchanged shared errors

The following lists every U cell with a shared arm error in at least one cohort. “All” means all four readers choose the same wrong side. “C1/C2 only” means both arms are wrong only in that cohort. See the exact source roster below for full context; no item was discarded for a small gap or awkward truncated ending.

| Cell | Wrong side → measured higher side | Error scope |
| --- | --- | --- |
| PC1 N05 | L → R | All four |
| PC1 N10 | L → R | All four |
| PC1 N15 | L → R | All four |
| PC1 N16 | R → L | All four |
| PC2 N01 | L → R | All four |
| PC2 N02 | R → L | All four |
| PC2 N05 | R → L | All four |
| PC2 N07 | L → R | All four |
| PC2 N08 | R → L | All four |
| PC2 N13 | L → R | All four |
| PC2 N14 | R → L | C2 only |
| PC3 N02 | R → L | All four |
| PC3 N07 | L → R | All four |
| PC3 N11 | L → R | All four |
| PC3 N13 | L → R | All four |
| PC4 N07 | L → R | All four |
| PC4 N14 | R → L | All four |
| PC4 N16 | L → R | C1 only |

These errors are not confined to an obscure direction. PC1 N05/N10/N15/N16 retain the research/list/time/transformation sides even when the corresponding platform/civilisation/club/event side is higher. PC2's all-reader errors cover both astronomy pairs N01/N02 and programming N05/N07/N08/N13; its football/ingredient reference contrast does not provide a verified rule for these within-category comparisons. PC3 N02/N07/N11/N13 retain seemingly investigation/behavior/security/structural alternatives but miss the frozen ordering. PC4 N07 and N14 are especially useful boundaries: all readers choose unspecified behavior and doneness respectively, yet copy-and-paste programming and pellet grills are higher. The PC4 N14 failure is therefore retained alongside the PC1 N14 repeat gain, rather than describing “measurement versus preparation” as a universal success. PC2 N14 and PC4 N16 add cohort disagreement without any arm difference.

These are observed shared mistakes, not evidence that E and EA used the same reasoning. Extra lists neither repaired these particular errors nor revealed a dependable confidence signal. Across all arm-specific agreements (including non-U cells), EA has 35 both-correct and 25 both-wrong versus E's 33 and 22: its greater total agreement does not establish better calibration or accuracy.

## Exact unchanged references

A is the original signed-direction token list; C is the original raw fit example. Both arms receive C, and only EA additionally receives A. JSON string quoting below preserves leading spaces, newlines, fragments and Unicode; array order is token rank. These are all 12 strings per pole, without selected translations. B was not displayed in this comparison and is not substituted here.

### PC1

```json
{
  "A": {
    "positive": [
      " photograph",
      "观测",
      "遥感",
      " imaging",
      "影像",
      " photographic",
      " photographed",
      " Observatory",
      " Telescope",
      " photographs",
      "探测",
      " Imaging"
    ],
    "negative": [
      "加薪",
      "垫",
      "打气",
      " chops",
      "打折",
      "让步",
      "王道",
      "菜",
      "wein",
      " shove",
      "蛋糕",
      "败"
    ]
  },
  "C": {
    "positive": "The spacecraft photographed",
    "negative": "A brief factual note:\nThe baker kneaded"
  }
}
```

### PC2

```json
{
  "A": {
    "positive": [
      "点球",
      "重创",
      "震慑",
      "连胜",
      "进球",
      " conced",
      " Yıld",
      "射门",
      " galard",
      " награ",
      "命中率",
      " awarded"
    ],
    "negative": [
      " recipes",
      " ingredients",
      " Ingredient",
      "ingredients",
      " ingredient",
      " mold",
      " kitchen",
      "ingredient",
      "杂货",
      " molds",
      " Ingredients",
      " Vegetable"
    ]
  },
  "C": {
    "positive": "A brief factual note:\nThe striker scored",
    "negative": "The baker kneaded"
  }
}
```

### PC3

```json
{
  "A": {
    "positive": [
      " testing",
      " evaluation",
      " Testing",
      "**:",
      " verification",
      " evaluations",
      " tests",
      " Evalu",
      " validation",
      " evaluating",
      " analyzing",
      " test"
    ],
    "negative": [
      "裹",
      "包裹",
      "身躯",
      " sacks",
      "stairs",
      " sandwich",
      "消磨",
      "捲",
      " squeez",
      " sandwiches",
      "缠绕",
      "缚"
    ]
  },
  "C": {
    "positive": "The developer tested",
    "negative": "A brief factual note:\nThe baker kneaded"
  }
}
```

### PC4

```json
{
  "A": {
    "positive": [
      " cameras",
      "目击",
      " UFO",
      "防震",
      " webcam",
      "拍到",
      "摄像",
      " glitches",
      "探测",
      "摄像机",
      "拍的",
      "幻影"
    ],
    "negative": [
      "调味品",
      "${",
      " Wealth",
      "调味",
      "情谊",
      "享",
      "薪",
      "津贴",
      "赠",
      "酬",
      " provision",
      " remuner"
    ]
  },
  "C": {
    "positive": "A brief factual note:\nThe goalkeeper blocked",
    "negative": "The recipe combined"
  }
}
```

## Exact 32-prefix roster

Rows are in frozen collection order, two adjacent sides per pair. Truncation and punctuation are unmodified; JSON `\\n`/quote escaping, where present, is representational only.

```json
[
  {
    "id": "N01-L",
    "topic": "astronomy",
    "prefix": "Pranav Sharma (Hindi: प्रणव शर्मा) is an astronomer and science historian known for his work on"
  },
  {
    "id": "N01-R",
    "topic": "astronomy",
    "prefix": "Andean dark constellations (Quechua:yana phuyu, lit. 'black cloud'; (Aymara:ch'iyar warawara) are astronomical configurations identified by pre-Columbian"
  },
  {
    "id": "N02-L",
    "topic": "astronomy",
    "prefix": "A synodic day (or synodic rotation period or solar day) is the period for a celestial"
  },
  {
    "id": "N02-R",
    "topic": "astronomy",
    "prefix": "Radio Galaxy Zoo (RGZ) is an internet crowdsourced citizen science project that seeks to locate supermassive"
  },
  {
    "id": "N03-L",
    "topic": "programming",
    "prefix": "A loop-switch sequence (also known as the for-case paradigm or Anti-Duff's Device) is a programming antipattern"
  },
  {
    "id": "N03-R",
    "topic": "programming",
    "prefix": "JuMP is an algebraic modeling language and a collection of supporting packages for mathematical optimization embedded"
  },
  {
    "id": "N04-L",
    "topic": "programming",
    "prefix": "Code Club is a voluntary initiative, founded in 2012. The initiative aims to provide opportunities for"
  },
  {
    "id": "N04-R",
    "topic": "programming",
    "prefix": "Floyd's triangle is a triangular array of natural numbers used in computer science education. It is"
  },
  {
    "id": "N05-L",
    "topic": "programming",
    "prefix": "The psychology of programming (PoP) is the field of research that deals with the psychological aspects"
  },
  {
    "id": "N05-R",
    "topic": "programming",
    "prefix": "Codecademy is an American online interactive platform that offers free coding classes in 13 different programming"
  },
  {
    "id": "N06-L",
    "topic": "programming",
    "prefix": "In software design, Procedural Design (SPD) converts and translates structural elements into procedural explanations. SPD starts"
  },
  {
    "id": "N06-R",
    "topic": "programming",
    "prefix": "In computing science and informatics, nesting is where information is organized in layers, or where objects"
  },
  {
    "id": "N07-L",
    "topic": "programming",
    "prefix": "In computer programming, unspecified behavior is behavior that may vary on different implementations of a programming"
  },
  {
    "id": "N07-R",
    "topic": "programming",
    "prefix": "Copy-and-paste programming, sometimes referred to as just pasting, is the production of highly repetitive computer programming"
  },
  {
    "id": "N08-L",
    "topic": "programming",
    "prefix": "The dangling else is a problem in programming of parser generators in which an optional else"
  },
  {
    "id": "N08-R",
    "topic": "programming",
    "prefix": "Performance portability refers to the ability of computer programs and applications to operate effectively across different"
  },
  {
    "id": "N09-L",
    "topic": "cooking",
    "prefix": "A pizza oven is an oven that is specially suited for making pizzas, especially Neapolitan pizza."
  },
  {
    "id": "N09-R",
    "topic": "cooking",
    "prefix": "Mise en place (French pronunciation: [mi zɑ̃ ˈplas]) is a French culinary phrase which means \"putting"
  },
  {
    "id": "N10-L",
    "topic": "astronomy",
    "prefix": "This is a list of galaxies sorted by surface brightness. Surface brightness is a measure of"
  },
  {
    "id": "N10-R",
    "topic": "astronomy",
    "prefix": "The astronomy of Meitei civilisation deals with celestial objects, space, and the physical universe as a"
  },
  {
    "id": "N11-L",
    "topic": "programming",
    "prefix": "Language-theoretic security, or LangSec, is an approach to software security that focuses on input handling, complexity,"
  },
  {
    "id": "N11-R",
    "topic": "programming",
    "prefix": "Sonic Pi is a free open-source live coding environment based on Ruby, originally designed to support"
  },
  {
    "id": "N12-L",
    "topic": "programming",
    "prefix": "In computer programming, a directive or pragma (from \"pragmatic\") is a language construct that specifies how"
  },
  {
    "id": "N12-R",
    "topic": "programming",
    "prefix": "Rosalind is an educational resource and web project for learning bioinformatics through problem solving and computer"
  },
  {
    "id": "N13-L",
    "topic": "programming",
    "prefix": "The Global Offset Table, or GOT, is a section of a computer program's (executables and shared"
  },
  {
    "id": "N13-R",
    "topic": "programming",
    "prefix": "In computing, hand coding means editing the underlying representation of a document or a computer program,"
  },
  {
    "id": "N14-L",
    "topic": "cooking",
    "prefix": "Pellet grills, sometimes referred to as pellet smokers, are outdoor cookers that combine elements of charcoal"
  },
  {
    "id": "N14-R",
    "topic": "cooking",
    "prefix": "Doneness is a gauge of how thoroughly cooked a cut of meat is based on its"
  },
  {
    "id": "N15-L",
    "topic": "programming",
    "prefix": "In computing, system time represents a computer system's notion of a point in time. System time"
  },
  {
    "id": "N15-R",
    "topic": "programming",
    "prefix": "Hack Club is a global nonprofit network of high school computer programming clubs founded in 2014"
  },
  {
    "id": "N16-L",
    "topic": "programming",
    "prefix": "An algorave (from an algorithm and rave) is an event where people dance to music generated"
  },
  {
    "id": "N16-R",
    "topic": "programming",
    "prefix": "The identity transform is a data transformation that copies the source data into the destination data"
  }
]
```

## What is still worth asking?

The defensible local use is **hypothesis generation about a signed contrast**: e.g., the unchanged PC4 list suggests a “glitches” association that the goalkeeper/recipe exemplars do not literally provide. The present data support consistent changes in comparisons, with both useful and adverse cases; they do not show reliable add-on prediction or a mechanism inside the readers. The fixed target is a population-derived coordinate at a 16-word final subtoken, not an external semantic label. Attributing a difference to token-induced anchoring would require separating reader variation and cue use; neither is observed here.

The smallest genuinely distinct future question is whether **fit-only contrast descriptions transfer better when covariance is fitted on a prespecified natural-text population matched to the intended endpoint**, rather than the original short authored action-prefix population. Define a modest disjoint fit/evaluation split and fit-only exemplars in advance; freeze any new directions before evaluation, and compare descriptions against those same examples. This would test the user's broader-corpus covariance idea, not rescue these old axes or repeat the display test until positive. It is a different basis/population question and needs its own bounded design; this note admits no acquisition, implementation or measurement. Merely repeating another authored panel or explaining away the harmful natural cases is less useful than consolidating the present boundary first.

Limitations retained: one small frozen model/layer, 16 text pairs, two same-family AI-reader cohorts, shared within-reader context and reader/order confounding; no causal cue attribution or equivalence claim. Exact first responses are audited, but encrypted prompt storage prevents an independent plaintext dispatch-byte audit. All retrospective semantic readings here are explicitly hypotheses, including the positive ones.

## Source pins and scope

- grades (artifact not distributed in this public snapshot): `4bae97cd90fb3de288a56624d919e57e739c90add2ac17519df5593a702dfe63`.
- dataset (artifact not distributed in this public snapshot): `d57300cb8511e409ff78e6ddc9fb6e1cf11b91bc629bf5492db8932cf585b129`.
- pairs (artifact not distributed in this public snapshot): `0ae85708e0b5da2942e31aa96af4ffa642c1294ff4c8b78612ec2dfc0ee54725`.
- references (artifact not distributed in this public snapshot): `0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c`.

All four SHA-256 pins were checked before JSON interpretation. The saved full-grade classifications were tabulated directly without importing or invoking a grader/checker/producer, reading a numerical archive, changing a scalar/sign/reference, using a reader, or contacting a source. The research-review skill guided evidence-first interpretation; the only new artifact is this requested report. Root owns durable synthesis and any future decision.
