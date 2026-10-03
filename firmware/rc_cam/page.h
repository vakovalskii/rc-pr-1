#pragma once
const char PAGE[] PROGMEM = R"rawliteral(<!doctype html>
<html lang="ru"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>RC-BUGGY cam</title>
<style>
:root{--bg:#0f1115;--panel:#161a23;--line:#2a3040;--fg:#e8e9ed;--dim:#8b93a1;--ok:#4ade80;--bad:#f87171;--acc:#60a5fa}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:13px/1.3 -apple-system,system-ui,sans-serif;overflow:hidden;touch-action:none;overscroll-behavior:none}
body{display:flex;flex-direction:column;padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)}
#bar{display:flex;align-items:center;gap:10px;padding:8px 12px;border-bottom:1px solid var(--line);white-space:nowrap;overflow:hidden}
#bar .s{color:var(--dim)} #bar b{color:var(--fg);font-variant-numeric:tabular-nums}
#dot{width:9px;height:9px;border-radius:50%;background:var(--bad);flex:none}
#state{font-weight:600}
#gear{margin-left:auto;background:none;border:1px solid var(--line);color:var(--fg);border-radius:8px;width:36px;height:32px;font-size:17px;flex:none}
#sticks{flex:1;display:flex;min-height:0;position:relative}
#cam{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;z-index:0}
.zone{z-index:1}
.base{background:rgba(22,26,35,.45)!important}
.zone{flex:1;position:relative;display:flex;align-items:center;justify-content:center;touch-action:none}
.zone+.zone{border-left:1px solid var(--line)}
.base{position:relative;border:2px solid var(--line);background:var(--panel);border-radius:999px}
#zT .base{width:96px;height:min(62vh,300px)}
#zS .base{height:96px;width:min(42vw,300px)}
.knob{position:absolute;left:50%;top:50%;width:76px;height:76px;margin:-38px 0 0 -38px;border-radius:50%;
  background:radial-gradient(circle at 35% 30%,#7db6ff,#3b82f6);box-shadow:0 4px 14px rgba(0,0,0,.5);transition:transform .12s}
.zone.live .knob{transition:none}
.cap{position:absolute;bottom:14px;left:0;right:0;text-align:center;color:var(--dim);font-size:12px}
.cap b{color:var(--fg);font-variant-numeric:tabular-nums}
#set{position:fixed;inset:auto 0 0 0;background:var(--panel);border-top:1px solid var(--line);padding:14px 14px calc(14px + env(safe-area-inset-bottom));
  transform:translateY(105%);transition:transform .2s;z-index:5}
#set.open{transform:none}
.row{display:flex;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.row label{color:var(--dim);min-width:92px}
input[type=range]{flex:1;min-width:140px;accent-color:var(--acc)}
button.b,select{background:#1d2230;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:9px 12px;font:inherit}
button.b:active{background:#2a3040}
.note{color:var(--dim);font-size:12px;margin-top:2px}
</style></head><body>
<div id="bar">
 <span id="dot"></span><span id="state">связь…</span>
 <span class="s">RTT <b id="rtt">—</b></span>
 <span class="s">сигнал <b id="rssi">—</b></span>
 <span class="s">ESC <b id="esc">—</b></span>
 <span class="s">видео <b id="fps">—</b> к/с · <b id="vms">—</b> мс</span>
 <button id="light" class="b" style="padding:4px 9px">фара</button>
 <button id="gear" aria-label="настройки">⚙</button>
</div>
<div id="sticks">
 <img id="cam" alt="">
 <div class="zone" id="zT"><div class="base"><div class="knob"></div></div><div class="cap">ГАЗ <b id="vT">0</b>%</div></div>
 <div class="zone" id="zS"><div class="base"><div class="knob"></div></div><div class="cap">РУЛЬ <b id="vS">0</b>%</div></div>
</div>
<div id="set">
 <div class="row"><label>газ максимум</label><input id="max" type="range" min="0" max="1000" step="50"><b id="maxv"></b></div>
 <div class="row"><label>разгон</label><input id="acc" type="range" min="100" max="3000" step="100"><b id="accv"></b></div>
 <div class="row"><label>торможение</label><input id="brk" type="range" min="100" max="4000" step="100"><b id="brkv"></b></div>
 <div class="row"><label>видео</label><select id="res"><option value="0">320×240 быстро</option><option value="1">480×320</option><option value="2">640×480 чётко</option></select></div>
 <div class="row"><label>качество</label><input id="q" type="range" min="8" max="40" step="2"><b id="qv"></b></div>
 <div class="row"><label>руль</label><button class="b" data-c="inv">инвертировать моторчик</button></div>
 <div class="note">руль без датчика положения: держишь стик — колёса повёрнуты</div>
 <div class="row" style="margin:12px 0 0"><button class="b" id="close" style="flex:1">готово</button></div>
</div>
<script>
const $=s=>document.querySelector(s);
let ws,steer=0,thr=0,id=0,sent=new Map(),rtt=null,maxSet=false;
function connect(){
 ws=new WebSocket('ws://'+location.hostname+':82/');
 ws.onopen=()=>{$('#dot').style.background='var(--ok)'};
 ws.onclose=()=>{$('#dot').style.background='var(--bad)';$('#state').textContent='нет связи';setTimeout(connect,700)};
 ws.onmessage=e=>{
  const p=e.data.split(',');if(p[0]!=='t')return;
  const [_,ack,esc,mot,fs,inv,max,src,acc,brk,efps,cl,res,q,rssi]=p;
  const t0=sent.get(+ack);if(t0!==undefined){const r=performance.now()-t0;rtt=rtt===null?r:rtt*.8+r*.2;$('#rtt').textContent=Math.round(rtt)+' мс';sent.clear()}
  const st=$('#state');
  if(fs==='1'){st.textContent='FAILSAFE';st.style.color='var(--bad)'}else{st.textContent='едем';st.style.color='var(--ok)'}
  $('#esc').textContent=esc;
  const r=+rssi;$('#rssi').textContent=r?r+' дБм':'—';$('#rssi').style.color=!r?'':r>-60?'var(--ok)':r>-75?'#fbbf24':'var(--bad)';
  if(!maxSet){$('#max').value=max;$('#maxv').textContent=Math.round(max/10)+'%';$('#acc').value=acc;$('#accv').textContent=(acc/1000).toFixed(1)+' с';$('#brk').value=brk;$('#brkv').textContent=(brk/1000).toFixed(1)+' с';if(document.activeElement!==$('#res'))$('#res').value=res;$('#q').value=q;$('#qv').textContent=(+q<=12?'выше':+q>=24?'ниже':'среднее');maxSet=true}
 };
}
connect();
const cam=$('#cam');let vms=null,vn=0,vt=performance.now();
async function video(){
 for(;;){
  if(document.hidden){await new Promise(r=>setTimeout(r,300));continue}
  const t0=performance.now();
  try{
   const r=await fetch('http://'+location.hostname+':81/jpg?'+t0,{cache:'no-store'});
   const u=URL.createObjectURL(await r.blob()),old=cam.src;cam.src=u;
   try{await cam.decode()}catch(e){}
   if(old.startsWith('blob:'))URL.revokeObjectURL(old);
   const dt=performance.now()-t0;vms=vms===null?dt:vms*.8+dt*.2;vn++;
  }catch(e){await new Promise(r=>setTimeout(r,500))}
 }
}
video();
setInterval(()=>{const now=performance.now();$('#fps').textContent=(vn*1000/(now-vt)).toFixed(0);vn=0;vt=now;if(vms!==null)$('#vms').textContent=Math.round(vms)},1000);
let lightOn=0;$('#light').onclick=()=>{lightOn^=1;send('light,'+lightOn);$('#light').style.background=lightOn?'#fbbf24':''};
let lastSend=0;
function sendCmd(){if(!ws||ws.readyState!==1)return;const i=++id;lastSend=performance.now();sent.set(i,lastSend);if(sent.size>40)sent.clear();
 ws.send('c,'+i+','+Math.round(steer*1000)+','+Math.round(thr*1000))}
const kick=()=>{if(performance.now()-lastSend>=20)sendCmd()};   // движение пальца уходит сразу, не ждёт тика
setInterval(sendCmd,50);                                       // пульс для failsafe
const send=s=>ws&&ws.readyState===1&&ws.send(s);
document.querySelectorAll('[data-c]').forEach(b=>b.onclick=()=>send(b.dataset.c));
$('#max').oninput=e=>{$('#maxv').textContent=Math.round(e.target.value/10)+'%';send('max,'+e.target.value)};
$('#res').onchange=e=>send('res,'+e.target.value);
$('#q').onchange=e=>send('q,'+e.target.value);
$('#acc').oninput=e=>{$('#accv').textContent=(e.target.value/1000).toFixed(1)+' с';send('acc,'+e.target.value)};
$('#brk').oninput=e=>{$('#brkv').textContent=(e.target.value/1000).toFixed(1)+' с';send('brk,'+e.target.value)};
$('#gear').onclick=()=>$('#set').classList.toggle('open');$('#close').onclick=()=>$('#set').classList.remove('open');
// джойстик: тянешь из любой точки зоны, ручка показывает отклонение; отпустил — в ноль
function stick(zone,axis,set,out){
 const base=zone.querySelector('.base'),knob=zone.querySelector('.knob');let pid=null,ox=0,oy=0;
 const range=()=>(axis==='y'?base.clientHeight:base.clientWidth)/2-38;
 const show=v=>{knob.style.transform=axis==='y'?`translateY(${-v*range()}px)`:`translateX(${v*range()}px)`;out.textContent=Math.round(v*100)};
 zone.addEventListener('pointerdown',e=>{pid=e.pointerId;ox=e.clientX;oy=e.clientY;zone.setPointerCapture(pid);zone.classList.add('live');e.preventDefault()});
 zone.addEventListener('pointermove',e=>{if(e.pointerId!==pid)return;
  const d=axis==='y'?-(e.clientY-oy):(e.clientX-ox);const v=Math.max(-1,Math.min(1,d/range()));set(v);show(v)});
 const up=e=>{if(e.pointerId!==pid)return;pid=null;zone.classList.remove('live');set(0);show(0)};
 zone.addEventListener('pointerup',up);zone.addEventListener('pointercancel',up);
 return show;
}
const showT=stick($('#zT'),'y',v=>{thr=v;kick()},$('#vT')),showS=stick($('#zS'),'x',v=>{steer=v;kick()},$('#vS'));
document.addEventListener('touchmove',e=>e.preventDefault(),{passive:false});
document.addEventListener('gesturestart',e=>e.preventDefault());
const K={ArrowUp:'u',KeyW:'u',ArrowDown:'d',KeyS:'d',ArrowLeft:'l',KeyA:'l',ArrowRight:'r',KeyD:'r'},held=new Set();
addEventListener('keydown',e=>{if(K[e.code]){held.add(K[e.code]);e.preventDefault()}});
addEventListener('keyup',e=>{held.delete(K[e.code]);if(!held.size){thr=0;steer=0;showT(0);showS(0)}});
setInterval(()=>{if(!held.size)return;thr=(held.has('u')?1:0)-(held.has('d')?1:0);steer=(held.has('r')?1:0)-(held.has('l')?1:0);showT(thr);showS(steer)},50);
</script></body></html>
)rawliteral";
