"""
Web Admin Dashboard — FastAPI + Jinja2 web UI.

Run: python api/dashboard.py
Open: http://localhost:8001
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from loguru import logger

from src.ocr.history_manager import HistoryManager
from src.review.queue_manager import ReviewQueueManager, CorrectionSubmission

app       = FastAPI(title="Document AI — Admin Dashboard")
templates = Jinja2Templates(directory=str(ROOT / "api" / "templates"))
history   = HistoryManager()
review_q  = ReviewQueueManager()

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Document AI — Admin</title>
<style>
  body{font-family:system-ui,sans-serif;margin:0;background:#f5f5f5;color:#333}
  .header{background:#1e3a5f;color:white;padding:16px 32px;display:flex;align-items:center;gap:12px}
  .header h1{margin:0;font-size:1.4rem}
  .nav{background:#fff;border-bottom:1px solid #ddd;padding:0 32px;display:flex;gap:0}
  .nav a{display:block;padding:12px 20px;text-decoration:none;color:#555;border-bottom:3px solid transparent}
  .nav a:hover,.nav a.active{color:#1e3a5f;border-bottom-color:#1e3a5f}
  .content{padding:32px;max-width:1200px;margin:0 auto}
  .card{background:#fff;border-radius:8px;padding:20px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
  .stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:24px}
  .stat{background:#1e3a5f;color:white;border-radius:8px;padding:16px;text-align:center}
  .stat .num{font-size:2rem;font-weight:700}
  .stat .lbl{font-size:.85rem;opacity:.8;margin-top:4px}
  table{width:100%;border-collapse:collapse;font-size:.9rem}
  th{background:#f8f8f8;text-align:left;padding:10px 12px;border-bottom:2px solid #ddd}
  td{padding:10px 12px;border-bottom:1px solid #eee;vertical-align:top}
  tr:hover td{background:#fafafa}
  .badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.75rem}
  .badge-ok{background:#d4edda;color:#155724}
  .badge-warn{background:#fff3cd;color:#856404}
  .badge-info{background:#d1ecf1;color:#0c5460}
  .preview{max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#666}
  button{background:#1e3a5f;color:white;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:.85rem}
  button:hover{background:#2a4f80}
  .empty{text-align:center;padding:40px;color:#999}
</style>
</head>
<body>
<div class="header">
  <span style="font-size:1.8rem">📄</span>
  <div>
    <h1>Document AI — Admin Dashboard</h1>
    <div style="font-size:.8rem;opacity:.8">Review queue & OCR history</div>
  </div>
</div>
<div class="nav">
  <a href="/" class="{active_home}">📊 Overview</a>
  <a href="/review" class="{active_review}">🔍 Review Queue</a>
  <a href="/history" class="{active_history}">📖 OCR History</a>
</div>
{content}
<script>
function approve(id){
  const val = document.getElementById('val_'+id).value;
  fetch('/api/review/'+id+'/approve',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({value:val,reviewer:'admin'})})
  .then(()=>location.reload());
}
</script>
</body>
</html>"""


def page(content: str, active: str = "home") -> HTMLResponse:
    html = HTML.format(
        content=content,
        active_home="active" if active == "home" else "",
        active_review="active" if active == "review" else "",
        active_history="active" if active == "history" else "",
    )
    return HTMLResponse(html)


@app.get("/", response_class=HTMLResponse)
async def overview():
    pending = review_q.pending_count()
    content = f"""
    <div class="content">
      <div class="stat-grid">
        <div class="stat"><div class="num">{pending}</div><div class="lbl">Pending Reviews</div></div>
      </div>
      <div class="card">
        <h3>Quick Links</h3>
        <p><a href="/review">🔍 Review Queue</a> — check low-confidence OCR results</p>
        <p><a href="/history">📖 OCR History</a> — browse past processed documents</p>
      </div>
    </div>"""
    return page(content, "home")


@app.get("/review", response_class=HTMLResponse)
async def review_page():
    items = review_q.fetch_pending(limit=50)
    if not items:
        rows = '<tr><td colspan="5" class="empty">✅ No items pending review</td></tr>'
    else:
        rows = ""
        for item in items:
            rows += f"""
            <tr>
              <td><code>{item.id[:8]}</code></td>
              <td>{item.field_name}</td>
              <td class="preview">{item.system_value}</td>
              <td><span class="badge badge-warn">{item.ocr_confidence:.0%}</span></td>
              <td>
                <input id="val_{item.id}" value="{item.system_value}" style="width:150px;padding:4px">
                <button onclick="approve('{item.id}')">✓ Approve</button>
              </td>
            </tr>"""

    content = f"""
    <div class="content">
      <div class="card">
        <h3>🔍 Review Queue ({len(items)} pending)</h3>
        <table>
          <tr><th>ID</th><th>Field</th><th>OCR Value</th><th>Conf</th><th>Action</th></tr>
          {rows}
        </table>
      </div>
    </div>"""
    return page(content, "review")


@app.get("/history", response_class=HTMLResponse)
async def history_page():
    from sqlalchemy import text as sql_text
    with history._Session() as session:
        from src.ocr.history_manager import HistoryRecord
        records = session.query(HistoryRecord).order_by(
            HistoryRecord.created_at.desc()
        ).limit(100).all()

    if not records:
        rows = '<tr><td colspan="5" class="empty">No history yet</td></tr>'
    else:
        rows = ""
        for r in records:
            preview = (r.full_text[:80] + "...") if len(r.full_text) > 80 else r.full_text
            rows += f"""
            <tr>
              <td><code>{r.doc_id}</code></td>
              <td>{r.user_id}</td>
              <td><span class="badge badge-info">{r.lang}</span></td>
              <td>{r.word_count}</td>
              <td class="preview" title="{r.full_text[:200]}">{preview}</td>
            </tr>"""

    content = f"""
    <div class="content">
      <div class="card">
        <h3>📖 OCR History ({len(records)} records)</h3>
        <table>
          <tr><th>Document</th><th>User</th><th>Lang</th><th>Words</th><th>Preview</th></tr>
          {rows}
        </table>
      </div>
    </div>"""
    return page(content, "history")


@app.post("/api/review/{item_id}/approve")
async def approve_review(item_id: str, body: dict):
    review_q.submit_correction(CorrectionSubmission(
        item_id=item_id,
        corrected_value=body.get("value", ""),
        reviewer_id=body.get("reviewer", "admin"),
    ))
    return {"status": "ok"}


if __name__ == "__main__":
    logger.info("Dashboard: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
