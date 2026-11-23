// Browser test of saved chats: create, follow up, new chat, reload, reopen, continue, delete.
// Start the stand-in model and the app first (see .github/workflows/ci.yml, job "ui"), then: npm test
import puppeteer from 'puppeteer-core';

const URL = process.env.RAG_URL || 'http://127.0.0.1:8770/';
const results = [];
const check = (name, ok, detail = '') => { results.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? ' — ' + detail : ''}`); };
const sleep = ms => new Promise(r => setTimeout(r, ms));

const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 900 });
const errors = [];
page.on('pageerror', e => errors.push('pageerror: ' + e.message));
page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });

async function ask(q) {
  const before = await page.$$eval('.msg', els => els.length);
  await page.click('#q', { clickCount: 3 });
  await page.type('#q', q);
  const done = page.waitForResponse(r => r.url().endsWith('/api/chat'), { timeout: 30000 });
  await page.keyboard.press('Enter');
  const res = await done;
  await res.text();                       // waits until the stream has finished
  await sleep(800);
  return { status: res.status(), added: (await page.$$eval('.msg', els => els.length)) - before };
}
const options = () => page.$$eval('#chats option', os => os.filter(o => o.value).map(o => ({ id: o.value, title: o.textContent })));
const messages = () => page.$$eval('.msg', els => els.map(e => ({ cls: e.className, text: e.querySelector('span')?.textContent || e.textContent })));

await page.goto(URL, { waitUntil: 'networkidle0' });
check('page loads without script errors', errors.length === 0, errors.join(' | '));
check('saved-chats menu hidden when there are no chats', await page.$eval('#chats', s => s.hidden));

let r = await ask('Which port does the Orion Billing service listen on?');
check('first question answered', r.status === 200 && r.added === 2, `status ${r.status}, ${r.added} messages added`);
let opts = await options();
check('conversation saved and listed', opts.length === 1 && !(await page.$eval('#chats', s => s.hidden)), JSON.stringify(opts));
check('title taken from the first question', /Orion Billing/i.test(opts[0]?.title || ''), opts[0]?.title);
check('delete button shown for the open chat', !(await page.$eval('#del', b => b.hidden)));

r = await ask('And which team owns it?');
opts = await options();
check('follow-up stays in the same conversation', opts.length === 1 && r.added === 2, `${opts.length} chats`);
await page.screenshot({ path: (process.env.SHOT_DIR || '/tmp') + '/1-conversation.png' });

await page.click('#clear');
check('New chat clears the screen', (await messages()).filter(m => /msg/.test(m.cls)).length === 0);
await ask('When is the production change freeze?');
opts = await options();
check('second conversation saved separately', opts.length === 2, JSON.stringify(opts.map(o => o.title)));

await page.reload({ waitUntil: 'networkidle0' });
opts = await options();
check('chats survive a page reload', opts.length === 2);
const first = opts.find(o => /Orion/i.test(o.title));
await page.select('#chats', first.id);
await sleep(800);
const restored = await messages();
const users = restored.filter(m => m.cls.includes('user')).map(m => m.text);
check('opening a saved chat restores every message', restored.length === 4 && users[0].includes('Orion') && users[1].includes('team owns'),
  `${restored.length} messages`);
check('restored answers keep their sources', (await page.$$('.msg.bot .src, .msg.bot details, .msg.bot .sources')).length > 0);
await page.screenshot({ path: (process.env.SHOT_DIR || '/tmp') + '/2-restored.png' });

r = await ask('How many approvals does payments-core need?');
const conv = await (await fetch(`${URL}api/conversations/${first.id}`)).json();
check('continuing a reopened chat appends to it', conv.length === 6, `${conv.length} stored messages`);

await page.click('#del');
await sleep(800);
opts = await options();
check('Delete removes the chat', opts.length === 1 && !opts.some(o => o.id === first.id));
check('screen resets after delete', (await messages()).length === 0);
const gone = await fetch(`${URL}api/conversations/${first.id}`);
check('deleted chat is gone on the server', gone.status === 404);

check('no script errors during the whole session', errors.length === 0, errors.join(' | '));
await page.screenshot({ path: (process.env.SHOT_DIR || '/tmp') + '/3-after-delete.png' });
await browser.close();
const failed = results.filter(r => !r.ok).length;
console.log(`\n${results.length - failed}/${results.length} passed`);
process.exit(failed ? 1 : 0);
