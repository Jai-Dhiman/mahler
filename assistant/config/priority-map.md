# Email Priority Map

Classification guide for Mahler email triage. Four tiers, listed in descending urgency.

---

## URGENT

Drop what you are doing and respond.

An email is URGENT only when it carries **both** an urgency signal **and** direct personal address. Bulk mail addressed to a list is never URGENT, no matter how strongly worded.

**Examples:**
- A person you know personally is asking for immediate help, a quick answer, or is clearly blocked waiting on you
- A system or service you are responsible for is down and someone is waiting on a fix
- A time-sensitive legal matter: court deadline, contract signature required, attorney follow-up
- A time-sensitive financial matter: wire transfer required, fraud alert, account suspension imminent
- A health matter requiring immediate action or response
- A direct reply thread where the other party has sent a follow-up indicating they are waiting
- An email with explicit urgency language ("urgent", "time-sensitive", "ASAP", "today", "deadline") from a real, known contact — not from a marketing tool or automated sender

---

## NEEDS_ACTION

Requires a response or decision, but not immediately. This is the default for any direct email that does not clearly fit URGENT, FYI, or NOISE. When uncertain, classify here — missing an important email is worse than seeing an extra one.

**Examples:**
- Meeting requests or scheduling coordination from a real person
- Direct questions that require a reply
- Code review requests or PR comments where you are the assigned reviewer
- GitHub pull request review requests, issue assignments, or direct @-mentions in comments
- Service account emails requiring an action: billing payment due, trial expiring, account at risk of suspension
- Personal emails from real people that do not fit URGENT but clearly expect a response
- Introductions or inbound inquiries that deserve a reply
- Any one-on-one email thread that is not clearly informational only

---

## FYI

Worth knowing, but no reply is needed. Read if interested; safe to skim or file immediately.

**Examples:**
- Bohemian Club (bohemianclub.org) — orchestra schedule updates, club communications, event announcements
- GitHub notifications for activity not assigned to you: new issues opened by others, PR merges on repos you watch, CI pass/fail results on branches you are not reviewing
- Shipping and delivery confirmations
- Bank statements, brokerage statements, credit card statements, and receipts
- Service status pages, incident postmortems, and changelog digests
- Emails where you are on CC and are not the primary recipient
- Calendar invitations you have already accepted (informational copy, no separate reply required)
- Automated system reports that are informational (cron summaries, analytics digests, usage reports)
- Acknowledgment emails ("thanks, received your message") that require no follow-up

---

## NOISE

Discard. No action, no reading, no storage beyond the triage log entry.

**Examples:**
- Any email with a `List-Unsubscribe` header or `Precedence: bulk` header (mailing list / bulk mail)
- Job alert emails from any job board or recruiting platform (LinkedIn, Indeed, Glassdoor, Hired, etc.)
- Marketing and promotional emails with unsubscribe links at the bottom
- Newsletter digests not actively read (tech roundups, product update newsletters, community digests)
- Emails from `noreply@`, `no-reply@`, `donotreply@`, or any address pattern indicating no human reads replies
- Social media notification digests (GitHub "catch up", Twitter/X digest, LinkedIn weekly)
- Automated password expiry reminders for services not in active use
- Survey and feedback request emails from SaaS tools

---

## Classification Rules

1. **Default to NEEDS_ACTION** when uncertain. It is better to surface an email that turns out to be FYI than to suppress one that needed a reply.
2. **URGENT requires both signals:** an urgency indicator AND direct personal address. A blast email with "URGENT" in the subject is still NOISE or NEEDS_ACTION based on the sender and content.
3. **FYI vs NEEDS_ACTION:** if a reply is clearly expected (question asked, action requested, thread awaiting response), it is NEEDS_ACTION. If it is purely informational with no implied obligation to respond, it is FYI.
4. **NOISE is for irrelevance at the source level:** if the sender or mailing list is structurally irrelevant (job boards, bulk marketing, automated no-reply), classify as NOISE without reading deeply.
5. **Known senders override defaults:** an email from a person in Jai's direct network (colleague, friend, family, collaborator) is presumed at least NEEDS_ACTION unless the content is purely informational.
