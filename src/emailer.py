"""Email digest — compile a high-value daily update with charts and the AI note.

Builds matplotlib charts (allocation, exposure vs target, value history, watchlist
momentum), composes an HTML email with the charts inline and the analyst note, and either
sends it via SMTP (Gmail by default) or renders a standalone local HTML preview.
"""
from __future__ import annotations

import base64
import os
import re
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import matplotlib
matplotlib.use("Agg")  # headless backend for scheduled runs
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import settings  # noqa: E402

GREEN, GREY, RED = "#2E7D32", "#9E9E9E", "#C62828"


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def build_charts(portfolio, exposure, history_df, screen_df, outdir: str) -> dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    charts: dict[str, str] = {}

    # Allocation
    w = pd.Series(portfolio.weights).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(w.index, w.values * 100, color=GREEN)
    ax.set_xlabel("% of account"); ax.set_title("Current allocation")
    p = os.path.join(outdir, "allocation.png"); _save(fig, p); charts["allocation"] = p

    # Exposure: current vs target
    ex = exposure.by_layer
    y = range(len(ex))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh([i + 0.2 for i in y], ex["current"] * 100, height=0.4, label="current", color=GREEN)
    ax.barh([i - 0.2 for i in y], ex["target"] * 100, height=0.4, label="thesis target", color=GREY)
    ax.set_yticks(list(y)); ax.set_yticklabels(ex.index)
    ax.set_xlabel("%"); ax.set_title("Exposure vs thesis target"); ax.legend()
    p = os.path.join(outdir, "exposure.png"); _save(fig, p); charts["exposure"] = p

    # Value history
    if history_df is not None and len(history_df) > 1:
        fig, ax = plt.subplots(figsize=(7.5, 3))
        ax.plot(history_df["ts"], history_df["total_value"], color=GREEN, marker="o", ms=3)
        ax.set_title("Portfolio value history"); ax.set_ylabel("$")
        fig.autofmt_xdate()
        p = os.path.join(outdir, "history.png"); _save(fig, p); charts["history"] = p

    # Watchlist 6M momentum
    if screen_df is not None and not screen_df.empty and "6M" in screen_df:
        s = screen_df.dropna(subset=["6M"]).sort_values("6M")
        if not s.empty:
            fig, ax = plt.subplots(figsize=(7.5, max(4, 0.25 * len(s))))
            colors = [RED if v < 0 else GREEN for v in s["6M"]]
            ax.barh(s["ticker"], s["6M"] * 100, color=colors)
            ax.set_xlabel("6-month return %"); ax.set_title("Thesis watchlist — 6M momentum")
            p = os.path.join(outdir, "watchlist.png"); _save(fig, p); charts["watchlist"] = p

    return charts


def _md_to_html(text: str) -> str:
    """Minimal markdown → HTML for the analyst note (headings, bullets, bold)."""
    if not text:
        return ""
    out, in_ul = [], False
    for raw in text.splitlines():
        line = raw.rstrip()
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        if re.match(r"^#{1,6}\s", line):
            if in_ul: out.append("</ul>"); in_ul = False
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{min(level + 1, 4)}>{line.lstrip('# ').strip()}</h{min(level + 1, 4)}>")
        elif re.match(r"^[-*]\s", line):
            if not in_ul: out.append("<ul>"); in_ul = True
            out.append(f"<li>{line[2:].strip()}</li>")
        elif not line.strip():
            if in_ul: out.append("</ul>"); in_ul = False
        else:
            if in_ul: out.append("</ul>"); in_ul = False
            out.append(f"<p>{line}</p>")
    if in_ul: out.append("</ul>")
    return "\n".join(out)


def compose_digest(portfolio, metrics, exposure, alerts, screen_df, history_df,
                   note_text, date_str, outdir: str = "notes/charts"):
    """Return (subject, html, charts: dict[cid->path])."""
    charts = build_charts(portfolio, exposure, history_df, screen_df, outdir)

    sev_color = {"HIGH": RED, "WARN": "#E65100", "INFO": "#1565C0"}
    alert_html = "".join(
        f'<li><span style="color:{sev_color[a.severity]};font-weight:bold">{a.severity}</span> '
        f'· {a.category} — {a.message}</li>' for a in alerts
    ) or "<li>No alerts.</li>"

    m = metrics
    metrics_html = "" if m is None else (
        f"1y return <b>{m.annualized_return*100:.1f}%</b> · "
        f"vol <b>{m.annualized_volatility*100:.0f}%</b> · "
        f"Sharpe <b>{m.sharpe:.2f}</b> · max DD <b>{m.max_drawdown*100:.0f}%</b>"
    )
    chart_order = ["allocation", "exposure", "history", "watchlist"]
    titles = {"allocation": "Allocation", "exposure": "Exposure vs thesis target",
              "history": "Value history", "watchlist": "Watchlist momentum"}
    img_html = "".join(
        f'<h3 style="margin:18px 0 4px">{titles[k]}</h3>'
        f'<img src="cid:{k}" style="max-width:680px;width:100%;border:1px solid #eee"/>'
        for k in chart_order if k in charts
    )

    subject = f"Portfolio Pilot — Daily Update ({date_str})"
    html = f"""<html><body style="font-family:Arial,Helvetica,sans-serif;color:#222;max-width:720px">
  <h2 style="margin-bottom:0">📈 Portfolio Pilot — {date_str}</h2>
  <p style="font-size:18px;margin:6px 0">
    <b>${portfolio.total_value:,.0f}</b>
    <span style="color:{GREEN if portfolio.total_pl>=0 else RED}">
      ({portfolio.total_pl_pct*100:+.1f}% / ${portfolio.total_pl:,.0f})</span>
  </p>
  <p style="color:#555">{metrics_html}</p>
  <h3>Alerts</h3><ul>{alert_html}</ul>
  {img_html}
  <h3 style="margin-top:24px">AI analyst note</h3>
  <div style="background:#fafafa;border:1px solid #eee;padding:12px 16px;border-radius:6px">
    {_md_to_html(note_text) or "<p><i>Note unavailable (no API key set).</i></p>"}
  </div>
  <p style="color:#999;font-size:12px;margin-top:18px">
    Decision support, not financial advice. Figures from yfinance; projections illustrative.</p>
</body></html>"""
    return subject, html, charts


def send_digest(subject: str, html: str, charts: dict[str, str]) -> None:
    """Send the digest via SMTP. Requires SMTP_PASS (Gmail App Password) in the env."""
    if not (settings.smtp_user and settings.smtp_pass and settings.email_to):
        raise RuntimeError(
            "Email not configured. Set SMTP_USER, SMTP_PASS (Gmail App Password) and "
            "EMAIL_TO in .env / secrets."
        )
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = settings.email_from or settings.smtp_user
    msg["To"] = settings.email_to
    alt = MIMEMultipart("alternative"); msg.attach(alt)
    alt.attach(MIMEText("Your Portfolio Pilot daily update — view in an HTML email client.", "plain"))
    alt.attach(MIMEText(html, "html"))
    for cid, path in charts.items():
        with open(path, "rb") as f:
            img = MIMEImage(f.read())
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=os.path.basename(path))
        msg.attach(img)
    with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port) as s:
        s.login(settings.smtp_user, settings.smtp_pass)
        s.send_message(msg)


def render_preview(html: str, charts: dict[str, str], out_path: str) -> str:
    """Write a standalone HTML preview with charts embedded as base64 data URIs."""
    h = html
    for cid, path in charts.items():
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        h = h.replace(f"cid:{cid}", f"data:image/png;base64,{b64}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(h)
    return out_path
