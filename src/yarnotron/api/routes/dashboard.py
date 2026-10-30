from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from yarnotron.api.auth import require_dashboard_login

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_class=HTMLResponse, dependencies=[Depends(require_dashboard_login)])
async def dashboard() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Yarnotron Parent Controls</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f7f4;
      color: #222;
    }
    body { margin: 0; }
    header {
      padding: 20px 28px;
      background: #24312f;
      color: #fff;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    h1 { font-size: 20px; margin: 0; font-weight: 650; letter-spacing: 0; }
    main {
      padding: 24px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 18px;
    }
    section {
      background: #fff;
      border: 1px solid #ddded8;
      border-radius: 8px;
      padding: 16px;
      min-height: 180px;
    }
    h2 { font-size: 16px; margin: 0 0 12px; }
    button, input, select, textarea {
      font: inherit;
      border-radius: 6px;
      border: 1px solid #bfc4bb;
      padding: 8px 10px;
      background: #fff;
    }
    button { cursor: pointer; background: #e8f0ee; }
    button.danger { background: #f7e3df; border-color: #d4aaa1; }
    button.secondary { background: #f4f5f1; }
    .row { display: flex; gap: 8px; margin-bottom: 12px; align-items: center; }
    .row input, .row select { flex: 1; min-width: 0; }
    .row input[type="checkbox"] { flex: 0 0 auto; width: 18px; height: 18px; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .item {
      border-top: 1px solid #eceee8;
      padding: 10px 0;
      font-size: 14px;
      line-height: 1.35;
    }
    .meta {
      color: #69716b;
      font-size: 12px;
      margin-top: 2px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      background: #edf2ef;
      border: 1px solid #d5ded8;
      color: #394640;
      font-size: 12px;
      margin-left: 6px;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 8px 0 0;
      font-size: 12px;
      background: #f4f5f1;
      padding: 8px;
      border-radius: 6px;
    }
    .empty, .status { color: #69716b; font-size: 13px; }
    .status { margin: 0; }
    .thumb {
      display: block;
      max-width: 100%;
      border: 1px solid #ddded8;
      border-radius: 6px;
      margin-top: 10px;
      background: #f4f5f1;
    }
  </style>
</head>
<body>
  <header>
    <h1>Yarnotron Parent Controls</h1>
    <button id="refresh-button" type="button">Refresh</button>
  </header>
  <main>
    <section>
      <h2>Stories</h2>
      <div class="row">
        <input id="story-title" placeholder="Story title">
      </div>
      <div class="row">
        <input id="story-synopsis" placeholder="Optional synopsis">
        <button id="create-story-button" type="button">Create</button>
      </div>
      <div id="stories"></div>
    </section>
    <section>
      <h2>Story Versions</h2>
      <div id="versions" class="empty">Select a story to review its saved versions.</div>
    </section>
    <section>
      <h2>Sessions</h2>
      <div id="sessions"></div>
    </section>
    <section>
      <h2>Recent Turns</h2>
      <div id="turns"></div>
    </section>
    <section>
      <h2>Voice Preferences</h2>
      <div class="row">
        <select id="voice-select"></select>
      </div>
      <label class="row">
        <input id="space-friend-howl-enabled" type="checkbox">
        <span>Space-friend howl for unclear story requests</span>
      </label>
      <div class="actions">
        <button id="sample-voice-button" type="button">Play sample</button>
        <button id="save-voice-button" type="button">Save preferences</button>
      </div>
      <div id="voice-meta" class="meta"></div>
    </section>
    <section>
      <h2>Safety Events</h2>
      <div id="safety"></div>
    </section>
    <section>
      <h2>Media Gallery</h2>
      <div id="media"></div>
    </section>
    <section>
      <h2>Memory Search</h2>
      <div class="row">
        <input id="memory-query" value="moonlight clock music">
        <button id="search-memory-button" type="button">Search</button>
      </div>
      <div id="memory" class="empty">Run a search to inspect and remove individual memories.</div>
    </section>
    <section>
      <h2>Action Log</h2>
      <p id="status" class="status">Ready.</p>
    </section>
  </main>
  <script>
    // When the server sets YARNOTRON_API_TOKEN every /v1 call needs it. Ask once
    // and keep it for this tab only (sessionStorage), never in a cookie or URL.
    const TOKEN_KEY = "yarnotron_api_token";
    const readToken = () => {
      try { return sessionStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; }
    };
    const json = async (url, options = {}) => {
      const send = () => {
        const headers = new Headers(options.headers || {});
        const token = readToken();
        if (token) {
          headers.set("Authorization", `Bearer ${token}`);
        }
        return fetch(url, {...options, headers});
      };
      let response = await send();
      if (response.status === 401) {
        const entered = window.prompt("This Yarnotron server needs its API token:");
        if (entered) {
          try { sessionStorage.setItem(TOKEN_KEY, entered.trim()); } catch {}
          response = await send();
        }
      }
      if (!response.ok) {
        const body = await response.text();
        throw new Error(`${response.status} ${body || response.statusText}`);
      }
      if (response.status === 204) {
        return null;
      }
      return response.json();
    };
	    const esc = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({
	      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
	    }[c]));
	    const date = (value) => value ? new Date(value).toLocaleString() : "open";
	    const actionHandlers = new Map();
	    let nextActionId = 0;

	    function registerAction(handler) {
	      const actionId = `action-${nextActionId}`;
	      nextActionId += 1;
	      actionHandlers.set(actionId, handler);
	      return actionId;
	    }

	    document.addEventListener("click", event => {
	      const target = event.target.closest("[data-action]");
	      if (!target) {
	        return;
	      }
	      const handler = actionHandlers.get(target.dataset.action);
	      if (handler) {
	        handler();
	      }
	    });

    function setStatus(message) {
      document.querySelector("#status").textContent = message;
    }

    async function runAction(message, action) {
      try {
        await action();
        setStatus(message);
      } catch (error) {
        setStatus(`Action failed: ${error.message}`);
      }
    }

    function item(title, meta, body, actions = "", badge = "", extra = "") {
      return `
        <div class="item">
          <strong>${esc(title)}</strong>${badge}
          <div class="meta">${esc(meta)}</div>
          ${body ? `<pre>${esc(body)}</pre>` : ""}
          ${extra}
          ${actions ? `<div class="actions">${actions}</div>` : ""}
        </div>`;
    }

	    function button(label, onClick, className = "secondary") {
	      const actionId = registerAction(onClick);
	      return `<button class="${esc(className)}" data-action="${actionId}" type="button">${esc(label)}</button>`;
	    }

	    async function loadStories() {
	      const stories = await json("/v1/stories");
	      document.querySelector("#stories").innerHTML = stories.map(story => {
	        const actions = [
	          button("Export", () => window.open(`/v1/stories/${story.id}/export.md`, "_blank")),
	          button("Versions", () => showVersions(story.id)),
	          button("Delete", () => deleteStory(story.id), "danger")
	        ].join("");
        const body = JSON.stringify({
          synopsis: story.synopsis,
          characters: story.characters?.map(c => ({
            name: c.name,
            role: c.role,
            traits: c.traits,
            notes: c.notes
          })),
          scenes: story.scenes?.map(s => ({
            title: s.title,
            sequence: s.sequence,
            summary: s.summary,
            draft_text: s.draft_text
          })),
          facts: story.facts?.map(f => f.text)
        }, null, 2);
        return item(story.title, `${story.id} · updated ${date(story.updated_at)}`, body, actions);
      }).join("") || '<div class="empty">No stories yet.</div>';
    }

    async function createStory() {
      const title = document.querySelector("#story-title").value.trim();
      const synopsis = document.querySelector("#story-synopsis").value.trim();
      if (!title) {
        setStatus("Story title is required.");
        return;
      }
      try {
        const story = await json("/v1/stories", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({title, synopsis: synopsis || null})
        });
        document.querySelector("#story-title").value = "";
        document.querySelector("#story-synopsis").value = "";
        await loadStories();
        setStatus(`Created ${story.title}. Use ${story.id} or the exact title to resume it later.`);
      } catch (error) {
        setStatus(`Action failed: ${error.message}`);
      }
    }

    async function deleteStory(storyId) {
      if (!confirm("Delete this story and its related memories? This cannot be undone.")) {
        return;
      }
      await runAction("Story deleted.", async () => {
        await json(`/v1/stories/${storyId}`, {method: "DELETE"});
        document.querySelector("#versions").innerHTML = "Select a story to review its saved versions.";
        document.querySelector("#memory").innerHTML = "Run a search to inspect and remove individual memories.";
        await Promise.all([loadStories(), loadSessions(), loadSafety(), loadMedia()]);
      });
    }

	    async function showVersions(storyId) {
	      await runAction("Loaded story versions.", async () => {
	        const versions = await json(`/v1/stories/${storyId}/versions`);
	        document.querySelector("#versions").innerHTML = versions.map(version => {
	          const actions = button("Restore", () => restoreVersion(storyId, version.id));
          const body = JSON.stringify({
            title: version.document.title,
            synopsis: version.document.synopsis,
            characters: version.document.characters?.map(c => c.name),
            scenes: version.document.scenes?.map(s => s.title)
          }, null, 2);
          return item(version.id, `saved ${date(version.created_at)}`, body, actions);
        }).join("") || '<div class="empty">No saved versions for this story yet.</div>';
      });
    }

    async function restoreVersion(storyId, versionId) {
      if (!confirm("Restore this saved story version? The current version will be saved first.")) {
        return;
      }
      await runAction("Story version restored.", async () => {
        await json(`/v1/stories/${storyId}/versions/${versionId}/restore`, {method: "POST"});
        await Promise.all([loadStories(), showVersions(storyId)]);
      });
    }

	    async function loadSessions() {
	      const sessions = await json("/v1/sessions");
	      const summaries = {};
	      await Promise.all(sessions.slice(0, 8).map(async session => {
	        try { summaries[session.id] = await json(`/v1/sessions/${session.id}/summary`); } catch (error) { /* shown without counts */ }
	      }));
	      document.querySelector("#sessions").innerHTML = sessions.map(session => {
	        const actions = session.ended_at ? "" : button("End session", () => endSession(session.id));
        const badges = [
          session.ended_at ? '<span class="badge">ended</span>' : '<span class="badge">active</span>',
          session.story_id ? "" : '<span class="badge">no story attached</span>'
        ].join("");
        const body = JSON.stringify({
          story_id: session.story_id,
          device_id: session.device_id,
          safety_summary: session.safety_summary,
          conversation: summaries[session.id]
        }, null, 2);
        return item(session.id, `${session.child_id} · ${date(session.started_at)} to ${date(session.ended_at)}`, body, actions, badges);
      }).join("") || '<div class="empty">No sessions yet.</div>';
    }

    async function loadTurns() {
      const turns = await json("/v1/conversation-turns?limit=20");
      document.querySelector("#turns").innerHTML = turns.map(turn => {
        const badge = turn.story_id ? `<span class="badge">${esc(turn.intent)}</span>` : `<span class="badge">${esc(turn.intent)}</span><span class="badge">not in story document</span>`;
        const body = JSON.stringify({
          story_id: turn.story_id,
          session_id: turn.session_id,
          child_said: turn.transcript,
          yarnotron_replied: turn.response_text,
          proposed_patches: turn.story_patches,
          applied_patches: turn.applied_story_patches,
          tutor_notes: turn.tutor_notes,
          safety: turn.safety
        }, null, 2);
        return item(turn.id, `${turn.child_id} · ${date(turn.created_at)}`, body, "", badge);
      }).join("") || '<div class="empty">No conversation turns yet.</div>';
    }

    async function endSession(sessionId) {
      await runAction("Session ended.", async () => {
        await json(`/v1/sessions/${sessionId}/end`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({})
        });
        await loadSessions();
      });
    }

    async function loadPreferences() {
      const [preferences, voices] = await Promise.all([
        json("/v1/preferences"),
        json("/v1/preferences/voices")
      ]);
      const select = document.querySelector("#voice-select");
      select.innerHTML = voices.map(voice =>
        `<option value="${esc(voice.id)}" data-provider="${esc(voice.provider)}">${esc(voice.name)}${voice.locale ? ` (${esc(voice.locale)})` : ""}</option>`
      ).join("");
      select.value = preferences.tts_voice || "";
      document.querySelector("#space-friend-howl-enabled").checked =
        preferences.space_friend_howl_enabled !== false;
      updateVoiceMeta();
    }

    function updateVoiceMeta() {
      const select = document.querySelector("#voice-select");
      const selected = select.options[select.selectedIndex];
      document.querySelector("#voice-meta").textContent =
        selected ? `Provider: ${selected.dataset.provider}` : "No voices discovered.";
    }

    function selectedVoicePreference() {
      const select = document.querySelector("#voice-select");
      const selected = select.options[select.selectedIndex];
      return {
        tts_provider: selected?.dataset.provider || "auto",
        tts_voice: select.value || null,
        space_friend_howl_enabled: document.querySelector("#space-friend-howl-enabled").checked
      };
    }

    async function sampleVoice() {
      await runAction("Voice sample played.", async () => {
        await json("/v1/preferences/voices/sample", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            ...selectedVoicePreference(),
            text: "Welcome to Yarnotron, what story would you like to work on?"
          })
        });
      });
    }

    async function saveVoicePreference() {
      await runAction("Voice preference saved.", async () => {
        await json("/v1/preferences", {
          method: "PUT",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(selectedVoicePreference())
        });
        await loadPreferences();
      });
    }

	    async function loadSafety() {
	      const events = await json("/v1/safety-events?limit=20");
	      document.querySelector("#safety").innerHTML = events.map(event => {
	        const reviewed = Boolean(event.reviewed_at);
	        const released = Boolean(event.released_at);
	        // A block stopped a turn the child never got back. Releasing re-runs
	        // it, so that one is offered until it has been used.
	        const releasable = ["block", "alert_parent"].includes(event.safety.action)
	          && !released
	          && Boolean(event.story_id)
	          && !(event.safety.categories || []).includes("session_ended");
	        const actions = [
	          releasable ? button("Release block", () => releaseSafety(event.id)) : "",
	          reviewed ? "" : button("Mark reviewed", () => reviewSafety(event.id))
	        ].join("");
        const badge = released
          ? '<span class="badge">released</span>'
          : reviewed ? '<span class="badge">reviewed</span>' : '<span class="badge">needs review</span>';
        const body = JSON.stringify({
          action: event.safety.action,
          categories: event.safety.categories,
          transcript_excerpt: event.transcript_excerpt,
          parent_alert_sent: event.parent_alert_sent,
          reviewed_at: event.reviewed_at,
          reviewed_by: event.reviewed_by,
          review_note: event.review_note,
          released_at: event.released_at,
          released_by: event.released_by
        }, null, 2);
        return item(event.safety.severity, event.created_at, body, actions, badge);
      }).join("") || '<div class="empty">No safety events.</div>';
    }

    async function reviewSafety(eventId) {
      const reviewNote = prompt("Optional parent note:");
      if (reviewNote === null) {
        return;
      }
      await runAction("Safety event marked reviewed.", async () => {
        await json(`/v1/safety-events/${eventId}/review`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({reviewed_by: "local_parent", review_note: reviewNote || null})
        });
        await loadSafety();
      });
    }

    async function releaseSafety(eventId) {
      if (!confirm("Release this block? The turn runs again and lands in the story, and the child is told about it next time they open it.")) {
        return;
      }
      const reviewNote = prompt("Why are you releasing it? (optional)");
      if (reviewNote === null) {
        return;
      }
      await runAction("Block released and the turn re-run.", async () => {
        await json(`/v1/safety-events/${eventId}/release`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({released_by: "local_parent", review_note: reviewNote || null})
        });
        await loadSafety();
        await loadStories();
      });
    }

	    async function loadMedia() {
	      const assets = await json("/v1/media?limit=20");
	      document.querySelector("#media").innerHTML = assets.map(asset => {
	        const actions = [];
	        if (asset.status === "pending_parent_review") {
	          actions.push(button("Approve", () => approveMedia(asset.id)));
	          actions.push(button("Reject", () => rejectMedia(asset.id), "danger"));
	        }
	        if (asset.status === "approved") {
	          actions.push(button("Generate", () => generateMedia(asset.id)));
	          actions.push(button("Reject", () => rejectMedia(asset.id), "danger"));
	        }
        const badge = `<span class="badge">${esc(asset.status)}</span>`;
        const image = asset.asset_url ? `<img class="thumb" alt="Generated story media" src="${esc(asset.asset_url)}">` : "";
        const body = JSON.stringify({
          prompt: asset.prompt,
          revised_prompt: asset.revised_prompt,
          safety: asset.safety,
          parent_approved_by: asset.parent_approved_by,
          parent_note: asset.parent_note,
          generator: asset.generator,
          failure_reason: asset.failure_reason
        }, null, 2);
        return item(asset.id, `${asset.child_id} · ${date(asset.created_at)}`, body, actions.join(""), badge, image);
      }).join("") || '<div class="empty">No media requests yet.</div>';
    }

    async function approveMedia(mediaId) {
      const parentNote = prompt("Optional parent note:");
      if (parentNote === null) {
        return;
      }
      await runAction("Media request approved.", async () => {
        await json(`/v1/media/${mediaId}/approve`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({reviewed_by: "local_parent", parent_note: parentNote || null})
        });
        await loadMedia();
      });
    }

    async function rejectMedia(mediaId) {
      const parentNote = prompt("Optional rejection note:");
      if (parentNote === null) {
        return;
      }
      await runAction("Media request rejected.", async () => {
        await json(`/v1/media/${mediaId}/reject`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({reviewed_by: "local_parent", parent_note: parentNote || null})
        });
        await loadMedia();
      });
    }

    async function generateMedia(mediaId) {
      await runAction("Media generated.", async () => {
        await json(`/v1/media/${mediaId}/generate`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({generator: "local_svg_placeholder"})
        });
        await loadMedia();
      });
    }

	    async function searchMemory() {
	      const query = document.querySelector("#memory-query").value;
	      const results = await json("/v1/memory/search", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({query, limit: 5})
	      });
	      document.querySelector("#memory").innerHTML = results.map(result => {
	        const actions = button("Remove memory", () => deleteMemory(result.id), "danger");
	        return item(result.id, `score ${Number(result.score).toFixed(3)} · ${date(result.created_at)}`, result.content, actions);
	      }).join("") || '<div class="empty">No matching memories.</div>';
	    }

    async function deleteMemory(memoryId) {
      if (!confirm("Remove this memory chunk from future searches?")) {
        return;
      }
      await runAction("Memory chunk removed.", async () => {
        await json(`/v1/memory/${memoryId}`, {method: "DELETE"});
        await searchMemory();
      });
    }

	    async function loadAll() {
	      await Promise.all([loadStories(), loadSessions(), loadTurns(), loadPreferences(), loadSafety(), loadMedia()]);
	      setStatus("Refreshed.");
	    }

	    document.querySelector("#refresh-button").addEventListener("click", loadAll);
	    document.querySelector("#create-story-button").addEventListener("click", createStory);
	    document.querySelector("#voice-select").addEventListener("change", updateVoiceMeta);
	    document.querySelector("#sample-voice-button").addEventListener("click", sampleVoice);
	    document.querySelector("#save-voice-button").addEventListener("click", saveVoicePreference);
	    document.querySelector("#search-memory-button").addEventListener("click", searchMemory);

	    loadAll().catch(error => setStatus(`Unable to load dashboard: ${error.message}`));
	  </script>
</body>
</html>
"""
