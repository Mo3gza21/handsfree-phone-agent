"""Simulated phone tasks with known-correct actions.

Each scenario is a goal plus a list of (screen, correct next action) steps.
Because the right answer is known, this data needs no LLM at all, and it is
also the yardstick the council uses to score each teacher model.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from spec import Element, Screen

# (english, arabic) display forms. A contact is stored in one script on the
# phone while the user may say it in the other, like on real phones.
NAMES = [
    ("Ahmed", "أحمد"), ("Mohamed", "محمد"), ("Sara", "سارة"), ("Mona", "منى"),
    ("Youssef", "يوسف"), ("Omar", "عمر"), ("Laila", "ليلى"), ("Khaled", "خالد"),
    ("Nour", "نور"), ("Mariam", "مريم"), ("Karim", "كريم"), ("Hana", "هنا"),
    ("Mostafa", "مصطفى"), ("Salma", "سلمى"), ("Tarek", "طارق"), ("Dina", "دينا"),
    ("Mom", "ماما"), ("Dad", "بابا"),
]
MESSAGES = [
    ("I'm on my way", "أنا في الطريق"), ("I'll be 10 minutes late", "هتأخر عشر دقايق"),
    ("Call me when you're free", "كلمني لما تفضى"), ("Happy birthday!", "كل سنة وانت طيب"),
    ("Did you arrive?", "وصلت؟"), ("I'm home", "أنا وصلت البيت"),
    ("See you tomorrow", "نشوفك بكرة"), ("Thank you", "شكرا"),
]
PLACES = [
    ("Cairo Festival City", "كايرو فستيفال سيتي"), ("Cairo Airport", "مطار القاهرة"),
    ("City Stars", "سيتي ستارز"), ("the nearest pharmacy", "أقرب صيدلية"),
    ("Mall of Egypt", "مول مصر"), ("Tahrir Square", "ميدان التحرير"),
    ("the nearest gas station", "أقرب بنزينة"), ("Zamalek", "الزمالك"),
]
QUERIES = [
    ("Amr Diab", "عمرو دياب"), ("Quran recitation", "قرآن كريم"), ("football highlights", "ملخص ماتش"),
    ("cooking molokhia", "طريقة عمل الملوخية"), ("relaxing music", "موسيقى هادية"),
    ("weather today", "الطقس النهارده"), ("Ahly match", "ماتش الأهلي"),
]
APPS = ["WhatsApp", "Phone", "Messages", "Settings", "Clock", "Maps", "YouTube", "Chrome",
        "Camera", "Gmail", "Calendar", "Photos", "Facebook", "Instagram", "Spotify", "Calculator"]
SNIPPETS = ["ok", "تمام", "See you", "👍", "Photo", "Voice message", "حاضر", "Where are you?", "Missed call"]
TIMES = ["7:00", "10:32", "Yesterday", "Monday", "9:15", "12:40"]


@dataclass
class Step:
    screen: Screen
    action: dict
    # Other actions that are just as right here (only used for scoring; training
    # always uses `action` so the student learns one consistent way).
    alts: list[dict] = field(default_factory=list)


@dataclass
class Scenario:
    family: str
    lang: str
    goal: str
    steps: list[Step]


class Builder:
    """Assigns element ids in reading order, like the app's screen reader does."""

    def __init__(self, app: str):
        self.screen = Screen(app)

    def add(self, role: str, label: str, **flags) -> Element:
        e = Element(len(self.screen.elements) + 1, role, label, **flags)
        self.screen.elements.append(e)
        return e


def _pick(rng, pairs, lang):
    p = rng.choice(pairs)
    return p, p[0] if lang == "en" else p[1]


def _say(lang, en, ar):
    return en if lang == "en" else ar


def start_screen(rng, target: str | None = None) -> Screen:
    """Whatever the user happened to be looking at."""
    if rng.random() < 0.5:
        b = Builder("Launcher")
        apps = rng.sample(APPS, 8)
        if target and target not in apps and rng.random() < 0.5:
            apps[rng.randrange(8)] = target
        for app in apps:
            b.add("button", app)
        b.add("edit", "Search")
        return b.screen
    b = Builder(rng.choice(["Instagram", "Facebook", "Gmail", "Calendar", "Photos"]))
    for _ in range(rng.randint(3, 6)):
        b.add(rng.choice(["text", "item", "button"]), rng.choice(SNIPPETS + ["Home", "Like", "Share", "Inbox"]))
    return b.screen


def open_step(rng, app) -> Step:
    screen = start_screen(rng, target=app)
    icon = [e for e in screen.elements if screen.app == "Launcher" and e.label == app]
    return Step(screen, {"action": "open_app", "app": app}, [{"action": "tap", "id": icon[0].id}] if icon else [])


def contact_name(rng, pair):
    return pair[0] if rng.random() < 0.6 else pair[1]


def chat_list(rng, target_label, include_target=True, contacts=None):
    b = Builder("WhatsApp")
    b.add("button", "Search")
    b.add("button", "More options")
    for t in ("Chats", "Updates", "Communities", "Calls"):
        b.add("tab", t, checked=(t == "Chats"))
    others = [contact_name(rng, p) for p in rng.sample(NAMES, 7) if target_label not in p]
    names = others[:rng.randint(4, 6)] if contacts is None else contacts
    if include_target:
        names = names[:]
        names.insert(rng.randint(0, len(names)), target_label)
    rows = {}
    for n in names:
        rows[n] = b.add("item", f"{n}, {rng.choice(SNIPPETS)}, {rng.choice(TIMES)}")
    b.screen.elements[-1].scrollable = False
    b.add("button", "New chat")
    return b.screen, rows


def chat_screen(name, typed=None, sent=None, rng=None):
    b = Builder("WhatsApp")
    b.add("button", "Navigate up")
    b.add("text", name)
    b.add("button", "Video call")
    b.add("button", "Voice call")
    for s in (rng.sample(SNIPPETS, 2) if rng else []):
        b.add("text", s)
    if sent:
        b.add("text", f"{sent}, Sent")
    box = b.add("edit", typed or "Message", focused=typed is not None)
    b.add("button", "Attach")
    b.add("button", "Camera")
    send = b.add("button", "Send") if typed else b.add("button", "Voice message")
    return b.screen, box, send


def whatsapp_message(rng, lang):
    pair, said_name = _pick(rng, NAMES, lang)
    msg_pair, msg = _pick(rng, MESSAGES, lang)
    goal = rng.choice({
        "en": [f"Send {said_name} a WhatsApp saying {msg}", f"WhatsApp {said_name}: {msg}",
               f"Tell {said_name} on WhatsApp \"{msg}\""],
        "ar": [f"ابعت لـ{said_name} على الواتساب {msg}", f"قول لـ{said_name} واتساب {msg}",
               f"ابعت رسالة واتساب لـ{said_name}: {msg}"],
    }[lang])
    stored = contact_name(rng, pair)
    steps = [open_step(rng, "WhatsApp")]
    if rng.random() < 0.3:
        s, _ = chat_list(rng, stored, include_target=False)
        steps.append(Step(s, {"action": "scroll", "direction": "down"}, [{"action": "tap", "id": 1}]))
    s, rows = chat_list(rng, stored)
    steps.append(Step(s, {"action": "tap", "id": rows[stored].id}))
    s, box, _ = chat_screen(stored, rng=rng)
    steps.append(Step(s, {"action": "type", "id": box.id, "text": msg}))
    s, _, send = chat_screen(stored, typed=msg, rng=rng)
    steps.append(Step(s, {"action": "tap", "id": send.id}))
    s, _, _ = chat_screen(stored, sent=msg, rng=rng)
    steps.append(Step(s, {"action": "done", "say": _say(lang, f"Sent to {said_name}.", f"اتبعتت لـ{said_name}.")}))
    return goal, steps


def read_last_message(rng, lang):
    pair, said_name = _pick(rng, NAMES, lang)
    stored = contact_name(rng, pair)
    last = rng.choice(MESSAGES)[rng.randint(0, 1)]
    goal = rng.choice({
        "en": [f"What did {said_name} send me?", f"Read me {said_name}'s last message"],
        "ar": [f"{said_name} بعتلي إيه؟", f"اقرالي آخر رسالة من {said_name}"],
    }[lang])
    steps = [open_step(rng, "WhatsApp")]
    s, rows = chat_list(rng, stored)
    steps.append(Step(s, {"action": "tap", "id": rows[stored].id}))
    b = Builder("WhatsApp")
    b.add("button", "Navigate up")
    b.add("text", stored)
    b.add("text", f"{rng.choice(SNIPPETS)}, Sent")
    b.add("text", f"{last}, {rng.choice(TIMES[:2] + TIMES[4:])}")
    b.add("edit", "Message")
    b.add("button", "Voice message")
    steps.append(Step(b.screen, {"action": "done", "say": _say(lang, f"{said_name} said: {last}", f"{said_name} قال: {last}")}))
    return goal, steps


def phone_contacts(rng, names, scroll_hint=False):
    b = Builder("Phone")
    b.add("button", "Search contacts")
    for t in ("Favorites", "Recents", "Contacts"):
        b.add("tab", t, checked=(t == "Contacts"))
    rows = {n: b.add("item", n) for n in names}
    b.add("button", "Keypad")
    return b.screen, rows


def call_contact(rng, lang):
    pair, said_name = _pick(rng, NAMES, lang)
    stored = contact_name(rng, pair)
    goal = rng.choice({
        "en": [f"Call {said_name}", f"Phone {said_name}", f"Can you call {said_name} for me"],
        "ar": [f"اتصل بـ{said_name}", f"كلم {said_name}", f"رن على {said_name}"],
    }[lang])
    steps = [open_step(rng, "Phone")]
    b = Builder("Phone")
    search = b.add("button", "Search contacts")
    tabs = {t: b.add("tab", t, checked=(t == "Recents")) for t in ("Favorites", "Recents", "Contacts")}
    alts = [{"action": "tap", "id": search.id}]
    for _ in range(3):
        who = contact_name(rng, rng.choice(NAMES))
        row = b.add("item", f"{who}, {rng.choice(['Missed', 'Outgoing', 'Incoming'])}, {rng.choice(TIMES)}")
        if who == stored:  # calling back from Recents works too
            alts.append({"action": "tap", "id": row.id})
    b.add("button", "Keypad")
    steps.append(Step(b.screen, {"action": "tap", "id": tabs["Contacts"].id}, alts))
    others = sorted({contact_name(rng, p) for p in NAMES if p != pair})
    if rng.random() < 0.35:  # target is further down the list
        first = rng.sample(others, 6)
        s, _ = phone_contacts(rng, first)
        steps.append(Step(s, {"action": "scroll", "direction": "down"}, [{"action": "tap", "id": 1}]))
    names = rng.sample(others, 5)
    names.insert(rng.randint(0, 5), stored)
    s, rows = phone_contacts(rng, names)
    steps.append(Step(s, {"action": "tap", "id": rows[stored].id}))
    b = Builder("Contacts")
    b.add("button", "Navigate up")
    b.add("text", stored)
    call = b.add("button", "Call")
    b.add("button", "Text")
    b.add("button", "Video")
    b.add("text", f"Mobile +20 10{rng.randint(10000000, 99999999)}")
    steps.append(Step(b.screen, {"action": "tap", "id": call.id}))
    b = Builder("Phone")
    b.add("text", stored)
    b.add("text", "Calling…")
    b.add("button", "Mute")
    b.add("button", "Speaker")
    b.add("button", "End call")
    steps.append(Step(b.screen, {"action": "done", "say": _say(lang, f"Calling {said_name}.", f"بتصل بـ{said_name}.")}))
    return goal, steps


def ambiguous_call(rng, lang):
    pair, said_name = _pick(rng, NAMES[:16], lang)
    a, c = f"{pair[0]} Work", f"{pair[0]} Gym"
    goal = _say(lang, f"Call {said_name}", f"اتصل بـ{said_name}")
    others = rng.sample([p[0] for p in NAMES if p != pair], 4)
    s, _ = phone_contacts(rng, sorted(others + [a, c]))
    q = _say(lang, f"Which {said_name}: {a} or {c}?", f"أنهي {said_name}: {a} ولا {c}؟")
    return goal, [open_step(rng, "Phone"), Step(s, {"action": "ask_user", "question": q})]


SETTINGS = {
    "wifi": ("Network & internet", "Wi-Fi", ("Wi-Fi", "الواي فاي")),
    "bluetooth": ("Connected devices", "Bluetooth", ("Bluetooth", "البلوتوث")),
    "dark": ("Display", "Dark theme", ("dark mode", "الوضع الليلي")),
    "airplane": ("Network & internet", "Airplane mode", ("airplane mode", "وضع الطيران")),
    "battery": ("Battery", "Battery Saver", ("battery saver", "توفير البطارية")),
}


def toggle_setting(rng, lang):
    key = rng.choice(list(SETTINGS))
    section, toggle, said = SETTINGS[key]
    thing = said[0] if lang == "en" else said[1]
    turn_on = rng.random() < 0.5
    already = rng.random() < 0.2
    goal = {
        ("en", True): rng.choice([f"Turn on {thing}", f"Enable {thing}", f"Switch {thing} on"]),
        ("en", False): rng.choice([f"Turn off {thing}", f"Disable {thing}"]),
        ("ar", True): rng.choice([f"شغل {thing}", f"افتح {thing}"]),
        ("ar", False): rng.choice([f"اقفل {thing}", f"طفي {thing}"]),
    }[(lang, turn_on)]
    steps = [open_step(rng, "Settings")]
    b = Builder("Settings")
    b.add("edit", "Search settings")
    sections = ["Network & internet", "Connected devices", "Apps", "Notifications", "Battery",
                "Storage", "Sound & vibration", "Display", "Security", "System"]
    rows = {s: b.add("item", s) for s in sections}
    steps.append(Step(b.screen, {"action": "tap", "id": rows[section].id}))
    current = turn_on if already else not turn_on

    def section_screen(state):
        b = Builder("Settings")
        b.add("button", "Navigate up")
        b.add("text", section)
        extras = {"Network & internet": ["Internet", "Calls & SMS", "SIMs", "Hotspot & tethering", "Wi-Fi", "Airplane mode"],
                  "Connected devices": ["Pair new device", "Bluetooth", "Connection preferences"],
                  "Display": ["Brightness level", "Dark theme", "Screen timeout", "Font size"],
                  "Battery": ["Battery usage", "Battery Saver", "Adaptive Battery"]}[section]
        sw = None
        for x in extras:
            if x == toggle:
                sw = b.add("switch", x, checked=state)
            elif x in ("Wi-Fi", "Airplane mode", "Bluetooth", "Dark theme", "Battery Saver", "Adaptive Battery"):
                b.add("switch", x, checked=rng.random() < 0.5)
            else:
                b.add("item", x)
        return b.screen, sw

    s, sw = section_screen(current)
    if already:
        msg = _say(lang, f"{thing} is already {'on' if turn_on else 'off'}.", f"{thing} {'شغال' if turn_on else 'مقفول'} أصلا.")
        steps.append(Step(s, {"action": "done", "say": msg}))
        return goal, steps
    steps.append(Step(s, {"action": "tap", "id": sw.id}))
    s, _ = section_screen(turn_on)
    msg = _say(lang, f"{thing} is {'on' if turn_on else 'off'}.", f"{thing} {'اشتغل' if turn_on else 'اتقفل'}.")
    steps.append(Step(s, {"action": "done", "say": msg}))
    return goal, steps


def set_alarm(rng, lang):
    h, m = rng.randint(4, 11), rng.choice([0, 15, 30, 45])
    ampm = rng.choice(["AM", "PM"])
    t = f"{h}:{m:02d} {ampm}"
    ar_ampm = "الصبح" if ampm == "AM" else "بالليل"
    goal = rng.choice({
        "en": [f"Set an alarm for {t}", f"Wake me up at {t}"],
        "ar": [f"اظبط منبه الساعة {h}:{m:02d} {ar_ampm}", f"صحيني الساعة {h}:{m:02d} {ar_ampm}"],
    }[lang])
    steps = [open_step(rng, "Clock")]

    def clock(tab):
        b = Builder("Clock")
        tabs = {x: b.add("tab", x, checked=(x == tab)) for x in ("Alarm", "Clock", "Timer", "Stopwatch")}
        return b, tabs

    first = rng.choice(["Clock", "Alarm"])
    if first == "Clock":
        b, tabs = clock("Clock")
        b.add("text", f"{rng.randint(1, 12)}:{rng.randint(10, 59)}")
        steps.append(Step(b.screen, {"action": "tap", "id": tabs["Alarm"].id}))
    b, _ = clock("Alarm")
    for _ in range(rng.randint(0, 3)):
        b.add("switch", f"{rng.randint(5, 9)}:{rng.choice(['00', '30'])} AM", checked=rng.random() < 0.5)
    add = b.add("button", "Add alarm")
    steps.append(Step(b.screen, {"action": "tap", "id": add.id}))
    b = Builder("Clock")
    b.add("text", "Select time")
    hour = b.add("edit", "Hour", focused=True)
    b.add("edit", "Minute")
    b.add("button", "AM", checked=True)
    b.add("button", "PM", checked=False)
    b.add("button", "Cancel")
    b.add("button", "OK")
    steps.append(Step(b.screen, {"action": "type", "id": hour.id, "text": str(h)}))
    b = Builder("Clock")
    b.add("text", "Select time")
    b.add("edit", str(h))
    minute = b.add("edit", "Minute", focused=True)
    am = b.add("button", "AM", checked=True)
    pm = b.add("button", "PM", checked=False)
    b.add("button", "Cancel")
    ok = b.add("button", "OK")
    steps.append(Step(b.screen, {"action": "type", "id": minute.id, "text": f"{m:02d}"}))
    b = Builder("Clock")
    b.add("text", "Select time")
    b.add("edit", str(h))
    b.add("edit", f"{m:02d}")
    am = b.add("button", "AM", checked=True)
    pm = b.add("button", "PM", checked=False)
    b.add("button", "Cancel")
    ok = b.add("button", "OK")
    if ampm == "PM":
        steps.append(Step(b.screen, {"action": "tap", "id": pm.id}))
        b = Builder("Clock")
        b.add("text", "Select time")
        b.add("edit", str(h))
        b.add("edit", f"{m:02d}")
        b.add("button", "AM", checked=False)
        b.add("button", "PM", checked=True)
        b.add("button", "Cancel")
        ok = b.add("button", "OK")
    steps.append(Step(b.screen, {"action": "tap", "id": ok.id}))
    b, _ = clock("Alarm")
    b.add("switch", t, checked=True)
    b.add("button", "Add alarm")
    steps.append(Step(b.screen, {"action": "done", "say": _say(lang, f"Alarm set for {t}.", f"المنبه اتظبط على {h}:{m:02d} {ar_ampm}.")}))
    return goal, steps


def navigate(rng, lang):
    pair, place = _pick(rng, PLACES, lang)
    goal = rng.choice({
        "en": [f"Take me to {place}", f"Navigate to {place}", f"Directions to {place}"],
        "ar": [f"وديني {place}", f"عايز أروح {place}", f"اتجاهات لـ{place}"],
    }[lang])
    steps = [open_step(rng, "Maps")]
    b = Builder("Maps")
    search = b.add("edit", "Search here")
    b.add("button", "Voice search")
    for x in ("Restaurants", "Gas", "Groceries", "Explore", "Go", "Saved", "Contribute"):
        b.add("button", x)
    steps.append(Step(b.screen, {"action": "type", "id": search.id, "text": place}))
    b = Builder("Maps")
    b.add("edit", place, focused=True)
    first = b.add("item", f"{pair[0]}, Cairo, Egypt")
    for x in rng.sample(PLACES, 2):
        if x != pair:
            b.add("item", f"{x[0]}, Cairo, Egypt")
    steps.append(Step(b.screen, {"action": "tap", "id": first.id}))
    b = Builder("Maps")
    b.add("text", pair[0])
    b.add("text", f"{rng.randint(5, 45)} min")
    d = b.add("button", "Directions")
    b.add("button", "Start")
    b.add("button", "Save")
    b.add("button", "Share")
    steps.append(Step(b.screen, {"action": "tap", "id": d.id}))
    b = Builder("Maps")
    b.add("text", f"Your location → {pair[0]}")
    for mode in ("Driving", "Transit", "Walking"):
        b.add("tab", mode, checked=(mode == "Driving"))
    start = b.add("button", "Start")
    steps.append(Step(b.screen, {"action": "tap", "id": start.id}))
    b = Builder("Maps")
    b.add("text", "Head north")
    b.add("button", "Exit navigation")
    steps.append(Step(b.screen, {"action": "done", "say": _say(lang, f"Navigating to {place}.", f"يلا بينا على {place}.")}))
    return goal, steps


def search_and_pick(rng, lang, app, query_pair, goal):
    q = query_pair[0] if lang == "en" else query_pair[1]
    steps = [open_step(rng, app)]
    b = Builder(app)
    if app == "YouTube":
        b.add("button", "Home")
        btn = b.add("button", "Search")
        for _ in range(3):
            b.add("item", f"{rng.choice(QUERIES)[0]} video, {rng.randint(1, 900)}K views")
        steps.append(Step(b.screen, {"action": "tap", "id": btn.id}))
        b = Builder(app)
        box = b.add("edit", "Search YouTube", focused=True)
        steps.append(Step(b.screen, {"action": "type", "id": box.id, "text": q}))
        b = Builder(app)
        b.add("edit", q, focused=True)
        sugg = [b.add("item", x) for x in (q, q + " live", q + " 2026")]
        steps.append(Step(b.screen, {"action": "enter"}, [{"action": "tap", "id": sugg[0].id}]))
        b = Builder(app)
        b.add("edit", q)
        first = b.add("item", f"{q} - official, {rng.randint(1, 50)}M views")
        b.add("item", f"{q} best moments, {rng.randint(1, 900)}K views")
        steps.append(Step(b.screen, {"action": "tap", "id": first.id}))
        b = Builder(app)
        b.add("button", "Pause")
        b.add("text", f"{q} - official")
        steps.append(Step(b.screen, {"action": "done", "say": _say(lang, f"Playing {q}.", f"شغلتلك {q}.")}))
    else:  # Chrome
        bar = b.add("edit", "Search or type web address")
        b.add("button", "Tabs")
        b.add("button", "More options")
        steps.append(Step(b.screen, {"action": "type", "id": bar.id, "text": q}))
        b = Builder(app)
        b.add("edit", q, focused=True)
        sugg = [b.add("item", x) for x in (q, q + " today")]
        steps.append(Step(b.screen, {"action": "enter"}, [{"action": "tap", "id": sugg[0].id}]))
        b = Builder(app)
        b.add("edit", q)
        b.add("text", "All")
        res = b.add("item", f"Top result for {q}")
        b.add("item", f"{q} - Wikipedia")
        steps.append(Step(b.screen, {"action": "done", "say": _say(lang, f"Here are the results for {q}.", f"دي نتايج {q}.")}))
    return goal, steps


def youtube(rng, lang):
    qp = rng.choice(QUERIES)
    q = qp[0] if lang == "en" else qp[1]
    goal = rng.choice({"en": [f"Play {q} on YouTube", f"Put on {q} on YouTube"],
                       "ar": [f"شغلي {q} على اليوتيوب", f"عايز أسمع {q} على يوتيوب"]}[lang])
    return search_and_pick(rng, lang, "YouTube", qp, goal)


def web_search(rng, lang):
    qp = rng.choice(QUERIES)
    q = qp[0] if lang == "en" else qp[1]
    goal = rng.choice({"en": [f"Google {q}", f"Search the web for {q}"],
                       "ar": [f"دور على {q} في جوجل", f"ابحث عن {q}"]}[lang])
    return search_and_pick(rng, lang, "Chrome", qp, goal)


def open_app(rng, lang):
    app = rng.choice(APPS)
    goal = rng.choice({"en": [f"Open {app}", f"Launch {app}", f"Go to {app}"],
                       "ar": [f"افتح {app}", f"افتحلي {app}", f"ودّيني على {app}"]}[lang])
    b = Builder(app)
    b.add("text", app)
    for _ in range(3):
        b.add("button", rng.choice(["Home", "Search", "Menu", "Settings", "Profile"]))
    return goal, [open_step(rng, app), Step(b.screen, {"action": "done", "say": _say(lang, f"{app} is open.", f"فتحتلك {app}.")})]


def take_photo(rng, lang):
    selfie = rng.random() < 0.5
    goal = {("en", True): "Take a selfie", ("en", False): "Take a photo",
            ("ar", True): "صورني سيلفي", ("ar", False): "صور صورة"}[(lang, selfie)]
    steps = [open_step(rng, "Camera")]

    def cam(front=False):
        b = Builder("Camera")
        b.add("text", "Front camera" if front else "Rear camera")
        b.add("button", "Settings")
        b.add("button", "Flash")
        for m in ("Night", "Portrait", "Photo", "Video"):
            b.add("tab", m, checked=(m == "Photo"))
        b.add("button", "Gallery")
        shutter = b.add("button", "Shutter")
        flip = b.add("button", "Switch camera")
        return b.screen, shutter, flip

    s, shutter, flip = cam()
    if selfie:
        steps.append(Step(s, {"action": "tap", "id": flip.id}))
        s, shutter, _ = cam(front=True)
    steps.append(Step(s, {"action": "tap", "id": shutter.id}))
    s, _, _ = cam(front=selfie)
    s.elements.insert(0, Element(0, "image", "Photo thumbnail, just now"))
    for i, e in enumerate(s.elements, 1):
        e.id = i
    steps.append(Step(s, {"action": "done", "say": _say(lang, "Photo taken.", "اتصورت.")}))
    return goal, steps


def sms(rng, lang):
    pair, said_name = _pick(rng, NAMES, lang)
    _, msg = _pick(rng, MESSAGES, lang)
    stored = contact_name(rng, pair)
    goal = rng.choice({"en": [f"Text {said_name} {msg}", f"Send an SMS to {said_name}: {msg}"],
                       "ar": [f"ابعت SMS لـ{said_name} {msg}", f"ابعت رسالة عادية لـ{said_name}: {msg}"]}[lang])
    steps = [open_step(rng, "Messages")]
    b = Builder("Messages")
    b.add("button", "Search")
    rows = {}
    names = [contact_name(rng, p) for p in rng.sample(NAMES, 5) if p != pair][:4]
    names.insert(rng.randint(0, len(names)), stored)
    for n in names:
        rows[n] = b.add("item", f"{n}, {rng.choice(SNIPPETS)}, {rng.choice(TIMES)}")
    b.add("button", "Start chat")
    steps.append(Step(b.screen, {"action": "tap", "id": rows[stored].id}))
    b = Builder("Messages")
    b.add("button", "Navigate up")
    b.add("text", stored)
    box = b.add("edit", "Text message")
    steps.append(Step(b.screen, {"action": "type", "id": box.id, "text": msg}))
    b = Builder("Messages")
    b.add("button", "Navigate up")
    b.add("text", stored)
    b.add("edit", msg, focused=True)
    send = b.add("button", "Send SMS")
    steps.append(Step(b.screen, {"action": "tap", "id": send.id}))
    b = Builder("Messages")
    b.add("text", stored)
    b.add("text", f"{msg}, Sent")
    steps.append(Step(b.screen, {"action": "done", "say": _say(lang, f"Text sent to {said_name}.", f"الرسالة اتبعتت لـ{said_name}.")}))
    return goal, steps


def go_back_or_home(rng, lang):
    home = rng.random() < 0.5
    goal = {("en", True): rng.choice(["Go home", "Go to the home screen"]),
            ("en", False): rng.choice(["Go back", "Back"]),
            ("ar", True): rng.choice(["روح للشاشة الرئيسية", "هوم"]),
            ("ar", False): rng.choice(["ارجع", "رجوع"])}[(lang, home)]
    s = start_screen(rng)
    act = {"action": "home"} if home else {"action": "back"}
    return goal, [Step(s, act), Step(start_screen(rng), {"action": "done", "say": _say(lang, "Done.", "تمام.")})]


FAMILIES = {
    "whatsapp_message": whatsapp_message, "read_last_message": read_last_message,
    "call_contact": call_contact, "ambiguous_call": ambiguous_call, "toggle_setting": toggle_setting,
    "set_alarm": set_alarm, "navigate": navigate, "youtube": youtube, "web_search": web_search,
    "open_app": open_app, "take_photo": take_photo, "sms": sms, "go_back_or_home": go_back_or_home,
}


def generate(n: int, seed: int = 0, weights: dict[str, float] | None = None, ar_ratio: float = 0.5):
    rng = random.Random(seed)
    names = list(FAMILIES)
    w = [(weights or {}).get(f, 1.0) for f in names]
    for _ in range(n):
        fam = rng.choices(names, w)[0]
        lang = "ar" if rng.random() < ar_ratio else "en"
        goal, steps = FAMILIES[fam](rng, lang)
        yield Scenario(fam, lang, goal, steps)
