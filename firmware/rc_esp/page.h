#pragma once
const char PAGE[] PROGMEM = R"rawliteral(<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>RC-BUGGY</title>
<style>
:root{--bg:#0f1115;--fg:#e8e9ed;--dim:#8b93a1;--ok:#4ade80;--bad:#f87171;--acc:#60a5fa}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;height:100%;overflow:hidden;background:var(--bg);color:var(--fg);font:13px/1.4 ui-monospace,Menlo,monospace;touch-action:none}
#hud{position:fixed;top:calc(env(safe-area-inset-top) + 8px);left:8px;right:8px;display:flex;gap:6px;flex-wrap:wrap;z-index:2;pointer-events:none}
.chip{background:#151922;border:1px solid #2a3040;border-radius:6px;padding:4px 8px}
.chip b{color:var(--acc)} .bad{color:var(--bad)} .ok{color:var(--ok)}
#pads{position:fixed;inset:0;display:flex}
.pad{flex:1;position:relative;border-right:1px dashed #1f2530}
.knob{position:absolute;width:76px;height:76px;margin:-38px 0 0 -38px;border-radius:50%;border:2px solid var(--acc);background:rgba(96,165,250,.16);opacity:0}
.pad.live .knob{opacity:1}
.lbl{position:absolute;bottom:calc(env(safe-area-inset-bottom) + 70px);width:100%;text-align:center;color:var(--dim)}
#set{position:fixed;left:8px;right:8px;bottom:calc(env(safe-area-inset-bottom) + 8px);display:flex;gap:6px;flex-wrap:wrap;z-index:2}
button,select{background:#151922;color:var(--fg);border:1px solid #2a3040;border-radius:6px;padding:7px 10px;font:inherit}
button:active{background:#2a3040} input[type=range]{width:110px;vertical-align:middle}
</style>
<div id="hud">
 <div class="chip">связь <b id="link">…</b></div>
 <div class="chip">RTT <b id="rtt">—</b> мс</div>
 <div class="chip" id="fs">—</div>
 <div class="chip">ESC <b id="esc">—</b> мкс</div>
 <div class="chip">потенц. <b id="pot">—</b></div>
 <div class="chip">моторчик <b id="mot">—</b></div>
 <div class="chip">калибровка <b id="cal">—</b></div>
</div>
<div id="pads">
 <div class="pad" id="padT"><div class="knob"></div><div class="lbl">ГАЗ — вверх/вниз</div></div>
 <div class="pad" id="padS"><div class="knob"></div><div class="lbl">РУЛЬ — влево/вправо</div></div>
</div>
<div id="set">
 <span class="chip">газ max <input id="max" type="range" min="0" max="1000" step="50"> <b id="maxv"></b></span>
 <select id="mode"><option value="0">руль: выкл</option><option value="1">руль: ручной</option><option value="2">руль: по потенциометру</option></select>
 <button data-c="cal,l">упор ←</button><button data-c="cal,c">центр</button><button data-c="cal,r">упор →</button>
 <button data-c="inv">инверт. моторчик</button>
</div>
<script>
const $=s=>document.querySelector(s);
let ws,steer=0,thr=0,id=0,sent=new Map(),rtt=null,maxSet=false;
function connect(){
 ws=new WebSocket('ws://'+location.hostname+':81/');
 ws.onopen=()=>{$('#link').textContent='ok';$('#link').className='ok'};
 ws.onclose=()=>{$('#link').textContent='нет';$('#link').className='bad';setTimeout(connect,700)};
 ws.onmessage=e=>{
  const p=e.data.split(','); if(p[0]!=='t')return;
  const [_,ack,pot,esc,mot,fs,mode,L,C,R,inv,max,src]=p;
  const t0=sent.get(+ack); if(t0!==undefined){const r=performance.now()-t0;rtt=rtt===null?r:rtt*.8+r*.2;$('#rtt').textContent=Math.round(rtt);sent.clear()}
  $('#fs').innerHTML=fs==='1'?'<span class="bad">FAILSAFE</span>':'<span class="ok">управление: '+src+'</span>';
  $('#esc').textContent=esc;$('#pot').textContent=pot;$('#mot').textContent=mot;
  $('#cal').textContent=(L<0||C<0||R<0)?'нет':L+' / '+C+' / '+R+(inv==='1'?' инв':'');
  if(document.activeElement!==$('#mode'))$('#mode').value=mode;
  if(!maxSet){$('#max').value=max;$('#maxv').textContent=Math.round(max/10)+'%';maxSet=true}
 };
}
connect();
setInterval(()=>{if(!ws||ws.readyState!==1)return;const i=++id;sent.set(i,performance.now());if(sent.size>40)sent.clear();
 ws.send('c,'+i+','+Math.round(steer*1000)+','+Math.round(thr*1000))},50);
const send=s=>ws&&ws.readyState===1&&ws.send(s);
document.querySelectorAll('button[data-c]').forEach(b=>b.onclick=()=>send(b.dataset.c));
$('#mode').onchange=e=>send('mode,'+e.target.value);
$('#max').oninput=e=>{$('#maxv').textContent=Math.round(e.target.value/10)+'%';send('max,'+e.target.value)};
function pad(el,axis,set){const k=el.querySelector('.knob');let pid=null,ox=0,oy=0;
 el.addEventListener('pointerdown',e=>{pid=e.pointerId;ox=e.clientX;oy=e.clientY;el.setPointerCapture(pid);el.classList.add('live');
  const r=el.getBoundingClientRect();k.style.left=(ox-r.left)+'px';k.style.top=(oy-r.top)+'px'});
 el.addEventListener('pointermove',e=>{if(e.pointerId!==pid)return;const d=axis==='y'?-(e.clientY-oy):(e.clientX-ox);set(Math.max(-1,Math.min(1,d/90)))});
 const up=e=>{if(e.pointerId!==pid)return;pid=null;el.classList.remove('live');set(0)};
 el.addEventListener('pointerup',up);el.addEventListener('pointercancel',up)}
pad($('#padT'),'y',v=>thr=v);pad($('#padS'),'x',v=>steer=v);
const K={ArrowUp:'u',KeyW:'u',ArrowDown:'d',KeyS:'d',ArrowLeft:'l',KeyA:'l',ArrowRight:'r',KeyD:'r'},held=new Set();
addEventListener('keydown',e=>{if(K[e.code]){held.add(K[e.code]);e.preventDefault()}});
addEventListener('keyup',e=>held.delete(K[e.code]));
setInterval(()=>{if(!held.size)return;thr=(held.has('u')?1:0)-(held.has('d')?1:0);steer=(held.has('r')?1:0)-(held.has('l')?1:0)},50);
addEventListener('keyup',()=>{if(!held.size){thr=0;steer=0}});
</script>
)rawliteral";
