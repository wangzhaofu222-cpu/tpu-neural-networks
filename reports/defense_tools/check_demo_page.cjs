// Offline component tests only; no browser, network or UI interaction.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=process.argv[2],html=fs.readFileSync(path.join(root,'实验过程展示.html'),'utf8');
const js=html.match(/<script>([\s\S]*)<\/script>/)[1];
const elements={};for(const m of html.matchAll(/id="([^"]+)"/g))elements[m[1]]={innerHTML:'',textContent:'',value:'1000',clientWidth:900};
const context2d=new Proxy({},{get:(_,key)=>()=>{}});elements.chart.getContext=()=>context2d;
const buttons=Array.from({length:10},()=>({classList:{toggle(){}}}));let callback=null;
const context={console,document:{getElementById:id=>{assert.ok(elements[id],id);return elements[id]},querySelectorAll:()=>buttons},window:{devicePixelRatio:1,addEventListener(){}},setInterval:fn=>{callback=fn;return 1},clearInterval:()=>{callback=null}};
vm.createContext(context);vm.runInContext(js+'\nglobalThis.app={choose,labs,draw};',context);
let checked=0;const links=new Set();
function linksIn(s){for(const m of s.matchAll(/(?:href|src)="([^"]+)"/g)){const p=decodeURIComponent(m[1]);if(p.startsWith('http'))throw Error('Unexpected network dependency');assert.ok(fs.existsSync(path.join(root,p)),p);links.add(p)}}
for(let n=0;n<10;n++){
 context.app.choose(n);assert.ok(elements.title.textContent.startsWith('Lab '+n+' '));
 for(const e of Object.values(elements))linksIn(e.innerHTML);
 assert.ok(elements.runs.innerHTML.includes(context.app.labs[n].model.evaluation_run));
 elements.progress.value='1';elements.progress.oninput();elements.play.onclick();assert.equal(typeof callback,'function');callback();elements.end.onclick();assert.equal(Number(elements.progress.value),1000);assert.equal(callback,null);
 checked++;
}
linksIn(html.replace(/<script>[\s\S]*<\/script>/,''));
const data={labs_tested:checked,unique_local_links_checked:links.size,all_targets_exist:true,javascript_execution:'PASS: ten selections, slider, replay, stop and full-curve handlers',network_dependencies:0,browser_visual_preview:'NOT_PERFORMED: browser URL policy disallows file://; no workaround attempted'};
fs.writeFileSync(path.join(root,'05_实验过程','页面功能核查.json'),JSON.stringify(data,null,2),'utf8');console.log(JSON.stringify(data,null,2));
