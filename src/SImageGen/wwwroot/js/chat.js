// Chat polling + composer helpers. Server renders initial thread; this only
// appends newly arrived messages and keeps the view pinned to the bottom.
(function () {
    const box = document.getElementById('messages');
    if (!box) return;
    const convoId = box.getAttribute('data-convo');
    const seen = new Set(
        Array.from(box.querySelectorAll('.msg')).map(el => el.getAttribute('data-id')));

    function scrollDown() { box.scrollTop = box.scrollHeight; }
    scrollDown();

    function bubble(role, text, imageUrl, status, error) {
        const wrap = document.createElement('div');
        wrap.className = 'msg';
        const b = document.createElement('div');
        if (role === 'user') {
            b.className = 'bubble user';
            b.textContent = text || '';
        } else {
            b.className = 'bubble assistant';
            if (imageUrl) {
                const img = document.createElement('img');
                img.src = imageUrl; img.loading = 'lazy'; img.alt = text || '';
                const cap = document.createElement('div');
                cap.className = 'cap';
                const span = document.createElement('span');
                span.textContent = text || '';
                const dl = document.createElement('a');
                dl.href = imageUrl; dl.target = '_blank'; dl.textContent = 'Download';
                dl.setAttribute('download', '');
                cap.appendChild(span); cap.appendChild(dl);
                b.appendChild(img); b.appendChild(cap);
            } else if (status === 'Running') {
                b.innerHTML = '<div class="spinner-border spinner-border-sm"></div>'
                    + '<span class="ms-2 text-muted">Working on your image…</span>';
            } else {
                const d = document.createElement('div');
                d.className = 'text-danger';
                d.textContent = 'Failed: ' + (error || 'unknown error');
                b.appendChild(d);
            }
        }
        wrap.appendChild(b);
        return wrap;
    }

    async function pollOnce() {
        if (!convoId) return false;
        const last = box.querySelector('.msg:last-child');
        const after = last ? last.getAttribute('data-at') : '';
        const r = await fetch('MessagesJson?id=' + encodeURIComponent(convoId)
            + '&after=' + encodeURIComponent(after || ''));
        if (!r.ok) return true;
        const items = await r.json();
        let running = false;
        for (const m of items) {
            if (seen.has(m.id)) continue;
            seen.add(m.id);
            const el = bubble(m.role, m.text, m.imageUrl, m.status, m.error);
            el.setAttribute('data-id', m.id);
            el.setAttribute('data-at', m.at);
            box.appendChild(el);
        }
        if (items.length) scrollDown();
        running = !!box.querySelector('.msg .spinner-border');
        return running;
    }

    async function loop() {
        try { if (!(await pollOnce())) return; } catch (e) { /* retry */ }
        setTimeout(loop, 2000);
    }
    if (convoId && box.querySelector('.spinner-border')) loop();

    // Enter = send, Shift+Enter = newline
    const ta = document.getElementById('promptField');
    const form = document.getElementById('composer');
    if (ta && form) {
        ta.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (ta.value.trim()) form.submit();
            }
        });
        ta.addEventListener('input', () => {
            ta.style.height = 'auto';
            ta.style.height = Math.min(ta.scrollHeight, 140) + 'px';
        });
    }

    // Attach filename preview
    const af = document.getElementById('attachField');
    const an = document.getElementById('attachName');
    if (af && an) {
        af.addEventListener('change', () => {
            an.textContent = af.files.length ? af.files[0].name : '';
        });
    }

    // New chat: clear thread client-side; Send creates the conversation server-side
    const nb = document.getElementById('newChatBtn');
    if (nb) {
        nb.addEventListener('click', () => {
            box.innerHTML = '<div class="empty-hero"><h3>What will you create?</h3>'
                + '<p class="text-muted">Describe an image — or attach a photo to restyle it.</p></div>';
            document.querySelectorAll('.convo-item').forEach(el => el.classList.remove('active'));
            const cf = document.getElementById('convoField');
            if (cf) cf.value = '';
            if (ta) ta.focus();
        });
    }
})();
