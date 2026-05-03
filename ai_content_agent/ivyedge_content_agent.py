"""
IvyEdge AI Content Agent
========================

A multi-step content generation pipeline that turns an editorial brief into a
publishable blog draft, while preserving IvyEdge brand voice.

Pipeline: Research -> Outline -> Draft -> Voice Edit -> SEO

Usage (programmatic):
    from ivyedge_content_agent import IvyEdgeContentAgent

    agent = IvyEdgeContentAgent()  # picks up ANTHROPIC_API_KEY from env
    result = agent.generate_blog_post(
        topic="How freelancers can prove income stability",
        persona="Maya",
        pillar="Pillar 1: Financial Education for Non-Traditional Paths",
        keywords=["freelance income proof", "1099 loan approval", "freelancer credit"],
    )

Outputs are returned as a dict with every intermediate step plus the final
draft. The CLI runner (run_pipeline.py) writes them to disk.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import anthropic
from dotenv import load_dotenv
from competitor_analysis import run_competitor_analysis

load_dotenv()

logger = logging.getLogger("ivyedge.agent")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("IVYEDGE_MODEL", "claude-sonnet-4-6")

# Token budgets per phase. Tune these against your typical output length.
PHASE_TOKEN_BUDGETS = {
    "research": 2500,
    "outline": 2500,
    "draft": 5000,
    "voice_edit": 5000,
    "seo": 5000,
    "social": 3000,
}

# Files in /context loaded on startup. Missing files are skipped with a warning
# rather than crashing — that lets you start with just brand_voice + personas
# and grow the library over time.
CORE_CONTEXT_FILES = {
    "brand_voice": "brand_voice.md",
    "personas": "personas.md",
    "product_knowledge": "product_knowledge.md",
    "strategy": "content_strategy.md",
    "inclusive_marketing": "inclusive_marketing.md",
}

# Folders walked recursively for additional context (research summaries,
# example articles). Each .md file is concatenated under its filename.
EXTRA_CONTEXT_DIRS = ["research_summaries", "examples"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ArticleBrief:
    """Editorial brief for a single blog post."""
    topic: str
    persona: str  # "Priya" | "Maya" | "Carmen" | "Dominique" | "All"
    pillar: str
    primary_keyword: str
    secondary_keywords: list[str] = field(default_factory=list)
    content_format: str = "educational"  # educational | customer_story | behavioral | industry
    target_word_count: tuple[int, int] = (1400, 1600)
    notes: str = ""

    @property
    def keyword_list(self) -> list[str]:
        return [self.primary_keyword] + self.secondary_keywords


@dataclass
class GenerationResult:
    """Full output of a generation run, including every intermediate step."""
    brief: ArticleBrief
    format_analysis: str = ""   # Phase 0 — competitive format benchmarks
    research: str = ""
    outline: str = ""
    first_draft: str = ""
    edited_draft: str = ""
    final_draft: str = ""
    social: str = ""
    meta_description: str = ""
    started_at: str = ""
    finished_at: str = ""
    model: str = DEFAULT_MODEL
    token_usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["brief"] = asdict(self.brief)
        return d


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class IvyEdgeContentAgent:
    """
    Five-phase content generation agent for the IvyEdge blog.

    The agent loads a context library (brand voice, personas, product knowledge,
    content strategy, plus any research summaries and example articles) and
    injects it into each phase's prompt. This keeps every draft on-brand
    without you having to re-paste guidelines into each request.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        context_dir: str | Path = "context",
    ):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY in your "
                "environment or pass api_key=... to IvyEdgeContentAgent()."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or DEFAULT_MODEL
        self.context_dir = Path(context_dir)
        self.context = self._load_context()
        self._cumulative_usage = {"input_tokens": 0, "output_tokens": 0}

    # -- Context loading --------------------------------------------------

    def _load_context(self) -> dict[str, str]:
        """Load all markdown context files into memory.

        Returns a dict keyed by short name (brand_voice, personas, ...) with
        the markdown contents. Also loads an `extras` key containing the
        concatenated contents of /research_summaries and /examples.
        """
        if not self.context_dir.exists():
            raise FileNotFoundError(
                f"Context directory not found: {self.context_dir.resolve()}\n"
                "Create it (with brand_voice.md, personas.md, etc.) before "
                "running the agent."
            )

        ctx: dict[str, str] = {}
        for key, filename in CORE_CONTEXT_FILES.items():
            path = self.context_dir / filename
            if path.exists():
                ctx[key] = path.read_text(encoding="utf-8")
            else:
                logger.warning("Missing context file: %s (skipping)", path)
                ctx[key] = ""

        # Walk the extra directories — any .md inside is concatenated into
        # one big block, with a header noting which file it came from.
        extras: list[str] = []
        for sub in EXTRA_CONTEXT_DIRS:
            sub_dir = self.context_dir / sub
            if not sub_dir.exists():
                continue
            for md in sorted(sub_dir.rglob("*.md")):
                extras.append(f"## --- {sub}/{md.name} ---\n\n{md.read_text(encoding='utf-8')}\n")
        ctx["extras"] = "\n".join(extras)

        logger.info(
            "Loaded context: %s",
            {k: f"{len(v)} chars" for k, v in ctx.items()},
        )
        return ctx

    def reload_context(self) -> None:
        """Reload context files from disk — handy when you've just edited a
        guideline doc and want the change picked up without restarting."""
        self.context = self._load_context()

    # -- Low-level call helper -------------------------------------------

    def _call_claude(self, prompt: str, max_tokens: int, phase: str) -> str:
        """Wrap the Anthropic SDK call with logging + retry."""
        for attempt in range(3):
            try:
                start = time.time()
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                elapsed = time.time() - start

                # Track usage so the CLI can report cost/tokens at the end
                if hasattr(msg, "usage") and msg.usage is not None:
                    self._cumulative_usage["input_tokens"] += getattr(msg.usage, "input_tokens", 0) or 0
                    self._cumulative_usage["output_tokens"] += getattr(msg.usage, "output_tokens", 0) or 0

                logger.info(
                    "[%s] %.1fs, in=%s out=%s",
                    phase,
                    elapsed,
                    getattr(msg.usage, "input_tokens", "?"),
                    getattr(msg.usage, "output_tokens", "?"),
                )
                return msg.content[0].text
            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                wait = 2 ** attempt
                logger.warning("[%s] %s — retrying in %ss: %s", phase, type(e).__name__, wait, e)
                time.sleep(wait)
        raise RuntimeError(f"[{phase}] Claude call failed after 3 retries")

    # -- Prompt assembly --------------------------------------------------

    def _voice_block(self) -> str:
        """The same voice reminder shows up in every phase."""
        return (
            "# IvyEdge brand voice (always)\n"
            "- Direct: lead with the answer; no hedging, no 'you might be wondering'.\n"
            "- Warm: acknowledge emotional reality; use contractions; say 'you'.\n"
            "- Grounded: tell the truth, even when uncomfortable; never over-promise.\n"
            "- Tagline to remember: 'Grow through anything.'\n\n"
            "## Words we use\n"
            "your money, your story, build, grow, here's how it works, "
            "you're in control, career gap, income pattern, trajectory, "
            "your full picture, no surprises\n\n"
            "## Words we avoid\n"
            "funds/monies, leverage, solutions, product suite, seamless, "
            "best-in-class, employment gap, unstable income, risk profile, "
            "tailored to your unique needs, please be advised\n\n"
            "## Voice calibration examples\n"
            "GOOD: 'Your 1099 income isn't unstable. Banks are measuring the wrong thing.'\n"
            "BAD:  'Freelance income may present challenges for traditional underwriting.'\n\n"
            "GOOD: 'Here's exactly what affects your credit score.'\n"
            "BAD:  'You might be wondering what impacts your credit score.'\n"
        )

    def _full_brand_context(self) -> str:
        """Pack the full context library into one block."""
        parts: list[str] = []
        for key in ("brand_voice", "personas", "product_knowledge", "strategy"):
            if self.context.get(key):
                parts.append(f"# === {key} ===\n\n{self.context[key]}")
        if self.context.get("extras"):
            parts.append(f"# === extra context ===\n\n{self.context['extras']}")
        return "\n\n".join(parts)

    # -- Phase 1: Research ------------------------------------------------

    def research_phase(self, brief: ArticleBrief) -> str:
        prompt = f"""You are a financial-services researcher preparing material for an IvyEdge blog post.

IMPORTANT — PRE-LAUNCH CONTEXT
IvyEdge has not launched any products yet. The blog exists to prove audience
demand for the IvyEdge thesis. Do not reference Ivy Smart Loan, Ivy Credit
Builder, Ivy Credit Monitor, Ivy Checking, or any other IvyEdge product as
if it exists. The goal is to demonstrate expertise on the topic and build
an audience — not to convert to a product.

ARTICLE BRIEF
- Topic: {brief.topic}
- Target persona: {brief.persona}
- Content pillar: {brief.pillar}
- Format: {brief.content_format}
- Primary keyword: {brief.primary_keyword}
- Secondary keywords: {", ".join(brief.secondary_keywords) or "(none)"}
- Notes from editor: {brief.notes or "(none)"}

RESEARCH TASKS
1. Identify 3-5 key insights about this topic that the target persona needs to know.
2. Surface relevant data points and statistics. Cite real, verifiable sources
   (Mintel, BLS, CFPB, Fed, peer-reviewed studies, major news outlets).
3. Name what traditional finance gets wrong about this topic.
4. Identify the perspective IvyEdge brings — the *point of view* on the topic,
   not a product pitch. Frame it as a thesis the reader can evaluate.
5. Suggest 1-2 anonymized examples or composite scenarios (NOT named member
   stories — we don't have members yet) that would resonate.

OUTPUT FORMAT (markdown)
## Key insights
- ...

## Relevant data
- ... (with source)

## Traditional approach (what's broken)
- ...

## IvyEdge angle
- ...

## Story or example ideas
- ...

{self._voice_block()}

# === Brand context ===
{self._full_brand_context()}
"""
        return self._call_claude(prompt, PHASE_TOKEN_BUDGETS["research"], "research")

    # -- Phase 2: Outline -------------------------------------------------

    def outline_phase(self, brief: ArticleBrief, research: str, format_guidance: str = "") -> str:
        format_block = (
            f"\nCOMPETITIVE FORMAT BENCHMARKS\n"
            f"The following analysis is based on the top free results for '{brief.primary_keyword}'.\n"
            f"Use it to set word count, heading structure, and section design.\n"
            f"Do not copy competitor angles — use this purely for structural guidance.\n\n"
            f"{format_guidance}\n"
        ) if format_guidance else ""

        prompt = f"""You are outlining an IvyEdge blog post.

ARTICLE BRIEF
- Topic: {brief.topic}
- Persona: {brief.persona}
- Pillar: {brief.pillar}
- Target length: {brief.target_word_count[0]}-{brief.target_word_count[1]} words
{format_block}
RESEARCH (from previous step)
{research}

PRE-LAUNCH CONTEXT
IvyEdge has not launched any products. CTAs are audience-building actions —
not product applications. Use one of:
  - Join the IvyEdge waitlist (be first when we launch)
  - Get the next post in your inbox (newsletter signup)
  - Tell us your story (collect demand signal)
  - Share this with someone who needs it (organic distribution)
  - Take our 2-minute survey on [topic-relevant question] (audience research)

OUTLINE REQUIREMENTS
- Structure: Hook -> Problem -> Insight / point of view -> Practical steps -> CTA
- 3-5 H2 sections, each with 2-3 H3 subsections where useful
- Each section should call out the specific data point or example to use
- End with a clear audience-building CTA from the list above

OUTPUT FORMAT
- Working title (one option, plus 2 alternates)
- Hook (2-3 sentences that name the problem directly)
- Section list with: H2 header / key points to cover / suggested example or stat
- Proposed CTA (specific audience-building action — waitlist, newsletter, share, survey)

VOICE REMINDER: lead with the answer. Be direct. Make it immediately useful.

{self._voice_block()}

# === Brand context ===
{self._full_brand_context()}
"""
        return self._call_claude(prompt, PHASE_TOKEN_BUDGETS["outline"], "outline")

    # -- Phase 3: Draft ---------------------------------------------------

    def draft_phase(self, brief: ArticleBrief, outline: str) -> str:
        prompt = f"""You are writing a blog post for IvyEdge based on this approved outline.

OUTLINE
{outline}

PRE-LAUNCH CONTEXT
IvyEdge has not launched any products. Do NOT reference Ivy Smart Loan, Ivy
Credit Builder, Ivy Credit Monitor, Ivy Checking, or any specific product.
Refer to IvyEdge as 'we' / 'us' (the perspective behind the post) — never as
a product the reader can apply for. The CTA must be an audience-building
action: waitlist, newsletter signup, share, survey, or 'tell us your story'.

WRITING GUIDELINES
- Voice: direct, warm, grounded. IvyEdge is the brilliant friend who happens
  to work in finance — not a bank, not a wellness app.
- Use 'you' and contractions naturally.
- Lead each section with the answer, then explain.
- Short paragraphs (3-4 sentences).
- Concrete examples and specific numbers from the research.
- Subheadings for scanability.
- This is thought leadership: prove we understand the topic and the reader's
  reality better than anyone else writing about it.

WHAT TO AVOID
- Generic financial advice ('make a budget')
- Jargon without explanation
- Talking down to readers
- Over-promising results ('transform your credit in 30 days')
- Passive voice and corporate speak
- Hedging ('may', 'might', 'could potentially')
- ANY mention of IvyEdge products as if they exist
- Phrases like 'apply today' or 'check your rate' — we have nothing to apply for

TARGET LENGTH: {brief.target_word_count[0]}-{brief.target_word_count[1]} words.

Return ONLY the blog post in clean markdown — no commentary, no wrappers.
Start with `# {{Working title}}` on the first line.

{self._voice_block()}

# === Brand context ===
{self._full_brand_context()}
"""
        return self._call_claude(prompt, PHASE_TOKEN_BUDGETS["draft"], "draft")

    # -- Phase 4: Voice edit ---------------------------------------------

    def voice_edit_phase(self, draft: str) -> str:
        prompt = f"""You are editing this IvyEdge blog draft to strengthen the brand voice.

DRAFT
{draft}

EDITING CHECKLIST
1. Replace jargon and corporate language with IvyEdge vocabulary.
2. Tighten the opening — does it lead with the answer?
3. Check for warmth — are we acknowledging emotional reality?
4. Remove hedging language ('may', 'might', 'could potentially').
5. Ensure contractions are used naturally.
6. Use 'you' — never 'borrowers', 'customers', 'consumers'.
7. Verify nothing over-promises ('guaranteed', 'transform', 'in 30 days').
8. Make sure practical steps are specific and actionable.
9. Vary sentence rhythm. Cut anything that sounds like a press release.
10. PRE-LAUNCH CHECK: Strip any reference to Ivy Smart Loan, Ivy Credit
    Builder, Ivy Credit Monitor, Ivy Checking, or 'apply' / 'check your
    rate' language. The CTA must be audience-building only (waitlist,
    newsletter, share, survey, tell us your story).

VOICE CALIBRATION
GOOD: "Your 1099 income isn't unstable. Banks are measuring the wrong thing."
BAD:  "Freelance income may present challenges for traditional underwriting."

GOOD: "This is frustrating — let's fix it."
BAD:  "We understand this situation may cause some concern."

OUTPUT
Return ONLY the revised post in clean markdown. No commentary, no diff —
the next phase needs a finished draft to pass to SEO.

{self._voice_block()}

# === Brand context ===
{self._full_brand_context()}
"""
        return self._call_claude(prompt, PHASE_TOKEN_BUDGETS["voice_edit"], "voice_edit")

    # -- Phase 5: SEO -----------------------------------------------------

    def seo_phase(self, brief: ArticleBrief, edited_draft: str) -> dict:
        """Returns a dict with keys: final_draft, meta_description,
        internal_link_suggestions, external_link_suggestions, alt_text_suggestions.
        Asks Claude to return JSON so we can parse cleanly."""

        prompt = f"""You are optimizing this IvyEdge blog post for SEO.

DRAFT
{edited_draft}

SEO TARGETS
- Primary keyword: {brief.primary_keyword}
- Secondary keywords: {", ".join(brief.secondary_keywords) or "(none)"}

SEO CHECKLIST
1. Integrate the primary keyword in the H1 title, the first 100 words, and at
   least one H2. Use it naturally 3-5 times across the body.
2. Weave secondary keywords in where they fit. Use semantic variations if a
   keyword feels forced.
3. Suggest 2-3 internal links (use placeholder anchor + IvyEdge URL slug
   like /products/ivy-smart-loan or /blog/{{slug}}).
4. Suggest 1-2 external links to authoritative sources (.gov, .edu, CFPB,
   Fed, BLS, peer-reviewed). Use real, plausible URLs you cite to in the
   draft itself.
5. Write a meta description: <=155 characters, includes primary keyword,
   action-oriented, value-forward.
6. Suggest alt text for any images the editor should add (descriptive +
   keyword where natural).

DO NOT sacrifice IvyEdge voice for keyword density. If the keyword doesn't
fit naturally, use a semantic variation.

OUTPUT FORMAT — two clearly separated sections, nothing else:

SECTION 1 — the full SEO-optimized post in markdown, between these exact delimiters:
===DRAFT_START===
<your markdown here>
===DRAFT_END===

SECTION 2 — metadata as a single valid JSON object, between these exact delimiters:
===META_START===
{{
  "meta_description": "<= 155 chars",
  "internal_link_suggestions": [
    {{"anchor_text": "...", "url": "/...", "where_in_post": "section name"}}
  ],
  "external_link_suggestions": [
    {{"anchor_text": "...", "url": "https://...", "source": "CFPB/Fed/...", "where_in_post": "..."}}
  ],
  "alt_text_suggestions": [
    {{"image_topic": "...", "alt_text": "..."}}
  ]
}}
===META_END===

{self._voice_block()}

# === Strategy context (SEO + pillars) ===
{self.context.get("strategy", "")}
"""
        raw = self._call_claude(prompt, PHASE_TOKEN_BUDGETS["seo"], "seo")
        return _parse_json_response(raw)

    # -- Phase 6: Social media --------------------------------------------

    @staticmethod
    def _substack_url(topic: str) -> str:
        slug = re.sub(r"[^a-z0-9\s-]", "", topic.lower())
        slug = re.sub(r"\s+", "-", slug).strip("-")[:80]
        return f"https://substack.com/@joinivyedge/p/{slug}"

    def social_phase(self, brief: ArticleBrief, final_draft: str) -> str:
        post_url = self._substack_url(brief.topic)
        prompt = f"""You are writing social media content to distribute an IvyEdge blog post.

PRE-LAUNCH CONTEXT
IvyEdge has not launched any products. Every CTA must be audience-building:
waitlist signup, newsletter, share, survey, or tell-us-your-story.
Never mention Ivy Smart Loan, Ivy Credit Builder, Ivy Credit Monitor, or Ivy Checking.

BLOG POST
Topic: {brief.topic}
Persona: {brief.persona}
Primary keyword: {brief.primary_keyword}
Substack URL: {post_url}

FULL FINAL DRAFT
{final_draft}

{self._voice_block()}

---

Produce all three assets below. Follow each format exactly.

---

## X / Threads

Write 3 post options (pick the strongest for posting, keep the others as alternates).
Rules:
- ≤ 280 characters each (including spaces and any hashtags)
- Lead with a punchy, opinionated first line — no throat-clearing
- One concrete insight or stat from the post
- End with the actual Substack URL ({post_url}) — not a placeholder like [link]
- 1–3 hashtags max; no more
- No em-dashes (—); use a dash (-) or a line break instead

Format:
### Option 1
<post text>

### Option 2
<post text>

### Option 3
<post text>

---

## Instagram

Write one caption + a suggested visual description.
Rules:
- Caption: 150–300 words; warm, direct IvyEdge voice; line breaks every 1–2 sentences
- Hook in the first line (no "Hey!" or emojis to open)
- 3–5 paragraphs; end with a question or CTA to drive comments; reference "link in bio" for the Substack post ({post_url})
- Hashtags: 10–15 highly relevant tags on a separate line at the bottom
- Visual: 1–2 sentences describing what the static image or carousel should show
  (use only IvyEdge brand colors from the brand voice guidelines; no invented colors)

Format:
### Caption
<caption text>

### Visual direction
<visual description>

---

## TikTok / Reels script

Write a complete, production-ready video script.
Rules:
- Length: 45–60 seconds of spoken content (roughly 120–160 words of dialogue)
- Hook: first 3 seconds must stop the scroll — a bold claim, surprising stat, or
  direct challenge to a common belief
- Structure: Hook → Problem (5 s) → 3 insights (25 s) → Payoff / CTA (10 s)
- Spoken dialogue only — no filler ("um", "so basically", "right?")
- On-screen text: include [TEXT: ...] cues for words to flash on screen
- B-roll / visual: include [VISUAL: ...] cues for what to show on camera
- CTA: end with one clear audience-building action; direct viewers to the Substack post at {post_url} (say "link in bio" for video, include the URL in the production notes)
- Tone: confident, knowledgeable friend — not a lecture, not a sales pitch

Format:
### Script

[HOOK - 0:00-0:03]
[TEXT: ...]
[VISUAL: ...]
<spoken line>

[PROBLEM - 0:03-0:08]
[VISUAL: ...]
<spoken lines>

[INSIGHT 1 - 0:08-0:18]
[TEXT: ...]
[VISUAL: ...]
<spoken lines>

[INSIGHT 2 - 0:18-0:28]
[TEXT: ...]
[VISUAL: ...]
<spoken lines>

[INSIGHT 3 - 0:28-0:38]
[TEXT: ...]
[VISUAL: ...]
<spoken lines>

[CTA - 0:38-0:48]
[TEXT: ...]
[VISUAL: ...]
<spoken lines>

### Production notes
<2–3 sentences on tone, setting, presenter energy, any props or graphics>
"""
        return self._call_claude(prompt, PHASE_TOKEN_BUDGETS["social"], "social")

    # -- Full pipeline ----------------------------------------------------

    def generate_blog_post(
        self,
        topic: str,
        persona: str,
        pillar: str,
        keywords: Iterable[str],
        content_format: str = "educational",
        notes: str = "",
        target_word_count: tuple[int, int] = (1400, 1600),
        on_phase: Optional[callable] = None,
    ) -> GenerationResult:
        """Run all five phases and return the assembled result.

        on_phase: optional callback fired with (phase_name, result_text) so
        callers can stream progress to a UI/log/Slack.
        """
        keywords = list(keywords)
        if not keywords:
            raise ValueError("At least one keyword is required (the primary).")

        brief = ArticleBrief(
            topic=topic,
            persona=persona,
            pillar=pillar,
            primary_keyword=keywords[0],
            secondary_keywords=keywords[1:],
            content_format=content_format,
            target_word_count=target_word_count,
            notes=notes,
        )

        result = GenerationResult(
            brief=brief,
            model=self.model,
            started_at=datetime.utcnow().isoformat() + "Z",
        )

        def step(name: str, fn):
            logger.info("---- Phase: %s ----", name)
            out = fn()
            if on_phase:
                on_phase(name, out)
            return out

        # Phase 0 — competitive format analysis (non-fatal if it fails)
        try:
            logger.info("---- Phase: format_analysis ----")
            _, guidance = run_competitor_analysis(brief.primary_keyword)
            result.format_analysis = guidance
            if on_phase:
                on_phase("format_analysis", guidance)
        except Exception as e:
            logger.warning("Format analysis skipped: %s", e)
            result.format_analysis = ""

        result.research = step("research", lambda: self.research_phase(brief))
        result.outline = step("outline", lambda: self.outline_phase(
            brief, result.research, result.format_analysis
        ))
        result.first_draft = step("draft", lambda: self.draft_phase(brief, result.outline))
        result.edited_draft = step("voice_edit", lambda: self.voice_edit_phase(result.first_draft))

        seo_out = step("seo", lambda: self.seo_phase(brief, result.edited_draft))
        result.final_draft = seo_out.get("final_draft", result.edited_draft)
        result.meta_description = seo_out.get("meta_description", "")

        result.social = step("social", lambda: self.social_phase(brief, result.final_draft))

        result.token_usage = {
            **self._cumulative_usage,
            "internal_link_suggestions": seo_out.get("internal_link_suggestions", []),
            "external_link_suggestions": seo_out.get("external_link_suggestions", []),
            "alt_text_suggestions": seo_out.get("alt_text_suggestions", []),
        }
        result.finished_at = datetime.utcnow().isoformat() + "Z"
        return result

    # -- Intro post (one-time founding statement) --------------------------

    def generate_intro_post(self, on_phase: Optional[callable] = None) -> GenerationResult:
        """Generate the IvyEdge founding/introduction post.

        This is a one-time brand story piece — shorter than a standard post,
        no keyword optimization, written as a direct letter to the reader.
        """
        brief = ArticleBrief(
            topic="Introducing IvyEdge",
            persona="All",
            pillar="Brand Story",
            primary_keyword="IvyEdge",
            content_format="brand_introduction",
            target_word_count=(700, 900),
            notes="Founding statement. Not keyword-optimized. Warm, personal, direct. Waitlist CTA.",
        )

        result = GenerationResult(
            brief=brief,
            model=self.model,
            started_at=datetime.utcnow().isoformat() + "Z",
        )

        def step(name: str, fn):
            logger.info("---- Phase: %s ----", name)
            out = fn()
            if on_phase:
                on_phase(name, out)
            return out

        prompt = f"""You are writing the founding/introduction post for IvyEdge — the very first thing
the world reads from us. This is not a blog post. It is a letter.

WHO WE ARE WRITING TO
All four of our personas at once: Priya (the career returner), Maya (the freelancer),
Carmen (the established entrepreneur), Dominique (the corporate climber). Each of them
has been doing everything right and still can't get a fair shot from traditional finance.

WHAT THIS POST MUST DO
1. Open with the problem — not our solution. The reader should feel seen before we say a word about ourselves.
2. Explain why the financial system fails these women (income type, career path, the metrics it uses).
3. Introduce IvyEdge — what we're building and why. One sentence on the mission.
4. Tell the reader what's coming: a blog that gives them the real information they've been denied,
   and products (launching soon) that evaluate their whole story.
5. End with a warm, direct CTA to join the waitlist (https://substack.com/@joinivyedge).

WHAT TO AVOID
- No corporate language. No "we're excited to announce." No "we're on a mission to."
- No product names — we haven't launched yet.
- Do not over-promise on the products. Say they're coming. Don't describe features.
- Do not write a listicle. This is prose.

TONE
The brilliant friend who happens to work in finance — the one you actually call.
She's been watching the system fail people she cares about and she's done being polite about it.
Warm. Direct. A little frustrated. Deeply hopeful.

TARGET LENGTH
700–900 words. No filler. Every sentence earns its place.

FORMAT
Return clean markdown only:
- H1 title (direct, human — not a tagline)
- 4–6 prose paragraphs
- A short, warm sign-off before the CTA
- CTA paragraph

{self._voice_block()}

# === Brand context ===
{self._full_brand_context()}
"""
        result.final_draft = step("intro", lambda: self._call_claude(
            prompt, 2000, "intro"
        ))

        # Social for the intro post
        result.social = step("social", lambda: self.social_phase(brief, result.final_draft))
        result.finished_at = datetime.utcnow().isoformat() + "Z"
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> dict:
    """Parse the SEO phase response, which uses delimited sections to keep
    the markdown draft separate from the JSON metadata."""
    draft = ""
    meta: dict = {}

    if "===DRAFT_START===" in raw and "===DRAFT_END===" in raw:
        draft = raw.split("===DRAFT_START===", 1)[1].split("===DRAFT_END===", 1)[0].strip()

    if "===META_START===" in raw and "===META_END===" in raw:
        meta_txt = raw.split("===META_START===", 1)[1].split("===META_END===", 1)[0].strip()
        # Strip any ```json fence the model might add
        if meta_txt.startswith("```"):
            meta_txt = meta_txt.split("\n", 1)[1] if "\n" in meta_txt else meta_txt
            if meta_txt.rstrip().endswith("```"):
                meta_txt = meta_txt.rstrip()[:-3]
        try:
            meta = json.loads(meta_txt)
        except json.JSONDecodeError as e:
            logger.warning("SEO metadata JSON invalid: %s", e)

    if not draft and not meta:
        logger.warning("SEO phase returned unexpected format — using raw output as draft")
        draft = raw

    return {
        "final_draft": draft or raw,
        "meta_description": meta.get("meta_description", ""),
        "internal_link_suggestions": meta.get("internal_link_suggestions", []),
        "external_link_suggestions": meta.get("external_link_suggestions", []),
        "alt_text_suggestions": meta.get("alt_text_suggestions", []),
    }
