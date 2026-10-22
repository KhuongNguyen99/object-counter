const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const zlib = require('node:zlib');
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const root = path.resolve(__dirname, '..');
const os = require('node:os');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'simple-counting-e2e-'));
function crc32(bytes) {
  let crc = -1;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
  }
  return (crc ^ -1) >>> 0;
}

function png(width = 1000, height = 800) {
  const chunk = (name, data) => {
    const kind = Buffer.from(name), size = Buffer.alloc(4), checksum = Buffer.alloc(4);
    size.writeUInt32BE(data.length);
    checksum.writeUInt32BE(crc32(Buffer.concat([kind, data])));
    return Buffer.concat([size, kind, data, checksum]);
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 6;
  const row = Buffer.alloc(width * 4 + 1, 255);
  row[0] = 0;
  for (let x = 0; x < width; x++) if (Math.floor(x / 8) % 2)
    row.fill(176, 1 + x * 4, 1 + x * 4 + 3);
  return Buffer.concat([
    Buffer.from("89504e470d0a1a0a", "hex"), chunk("IHDR", header),
    chunk("IDAT", zlib.deflateSync(Buffer.concat(Array.from({ length: height }, () => row)))),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}


const out = process.env.E2E_OUTPUT || path.join(root, 'test-results/simple-counting/browser');
const photo = path.join(temp, 'parts.png');
fs.writeFileSync(photo, png());
const fixture = {width:1000,height:800,boxes:Array.from({length:40},(_,i)=>{const x=80+(i%10)*90,y=70+Math.floor(i/10)*170;return [x,y,x+40,y+30];})};
let session, submitted, detects = 0;
const uploads = [];
const catalog = { default_model:'countgd-plusplus', models:[{id:'countgd-plusplus', name:'CountGD++',available:true,
  reference_mode:'per_image',required_reference_boxes:10,default_confidence:.8}],device:'cuda' };
const result = {id:'photo',filename:'parts.jpg',count:14,width:1000,height:800,reviewed:false,review_revision:0,
  elapsed_ms:1234,processing:{tiles:12,tile_size:800,duplicates_removed:10},warning:'Experimental CountGD++ model',
  detections:[{index:1,box:[200,200,260,260],confidence:.85}],original_url:'/photo.jpg',annotated_url:'/photo.jpg'};
function json(res, data, status=200) {res.writeHead(status,{'content-type':'application/json'});res.end(JSON.stringify(data));}
async function body(req) {const chunks=[];for await(const chunk of req)chunks.push(chunk);return Buffer.concat(chunks).toString('latin1');}
function field(input,name) {return new RegExp('name="'+name+'"\\r\\n\\r\\n([^\\r]*)').exec(input)?.[1];}
const server = http.createServer(async(req,res)=>{
  try {
    const url = req.url.split('?')[0];
    if(url==='/api/models')return json(res,catalog);
    if(url==='/api/reference-boxes') {await body(req); detects++;return json(res,fixture);}
    if(url==='/api/sessions' && req.method==='POST') {
      const data=await body(req);submitted={model:field(data,'model'),confidence:field(data,'confidence'),label:field(data,'label')};
      session={id:'simple',label:'',model:submitted.model,reference_mode:'per_image',created_at:'2026-10-22T12:00:00Z',status:'active',total_count:0,image_count:0,references:[],batches:[]};
      return json(res,session,201);
    }
    if(url==='/api/sessions/simple/batches') {
      const data=await body(req),mode=field(data,'count_mode') || 'standard';
      submitted.boxes=JSON.parse(field(data,'image_boxes') || '[]');
      submitted.searchAreas=JSON.parse(field(data,'search_areas') || '[]');
      submitted.countMode=mode;
      uploads.push(structuredClone(submitted));
      const counted=mode==='quick'?{...result,id:'quick-photo',count:3,warning:null,detections:[
        {index:1,box:[120,130,160,160],confidence:null},{index:2,box:[220,130,260,160],confidence:null},
        {index:3,box:[320,130,360,160],confidence:null}]}:result;
      session.batches.push({id:'batch-'+session.batches.length,status:'complete',count_mode:mode,total_count:counted.count,results:[counted]});
      session.total_count=session.batches.reduce((total,batch)=>total+batch.total_count,0);session.image_count=session.batches.length;
      return json(res,{id:'batch',status:'complete'},202);
    }
    if(url==='/api/sessions/simple')return json(res,session);
    if(url==='/api/sessions/simple/images/photo/review') {
      const data=JSON.parse(await body(req));result.reviewed=data.reviewed;result.review_revision++;result.count=data.boxes.length;
      result.detections=data.boxes.map((box,index)=>({box,index:index+1}));session.total_count=result.count;
      return json(res,session);
    }
    const file=url==='/photo.jpg'?photo:path.join(root,'app/static',url==='/'?'index.html':url.slice(1));
    res.writeHead(200,{'content-type':{'.html':'text/html','.js':'text/javascript','.css':'text/css','.jpg':'image/jpeg','.png':'image/png','.svg':'image/svg+xml'}[path.extname(file)]});
    fs.createReadStream(file).pipe(res);
  }catch(error){json(res,{detail:error.message},500);}
});
(async()=>{
  fs.mkdirSync(out,{recursive:true});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({executablePath:process.env.EDGE_EXE || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
    page.setDefaultTimeout(20000);
    page.on('pageerror',e=>errors.push(e.message));
    const base='http://127.0.0.1:'+server.address().port;
    await page.goto(base);
    await page.waitForFunction(()=>!document.querySelector('#start-button').disabled);
    assert.equal(await page.locator('#model-select').isVisible(),false);
    assert.equal(await page.locator('#confidence').inputValue(),'0.8');
    await page.locator('.advanced summary').click();
    await page.screenshot({path:path.join(out,'start-desktop.png'),fullPage:true});
    const noJargon=async()=>assert.doesNotMatch(await page.locator('body').innerText(),/CountGD|YOLO|SAM3|FastSAM|model|confidence|proposal|coordinates|tile|\bpx\b|GPU|CUDA|experimental/i);
    await noJargon();
    await page.locator('#target-gallery').setInputFiles(photo);
    await page.locator('#editor-image').waitFor();
    await page.waitForFunction(()=>document.querySelector('#editor-image').complete && document.querySelector('#editor-image').naturalWidth>0);
    assert.equal(detects,0);
    await page.locator('#reference-select-areas').click();
    await page.locator('#box-overlay').scrollIntoViewIfNeeded();
    let bounds=await page.locator('#box-overlay').boundingBox();
    await page.mouse.move(bounds.x+.001,bounds.y+.001);await page.mouse.down();
    await page.mouse.move(bounds.x+bounds.width-.001,bounds.y+bounds.height-.001,{steps:12});await page.mouse.up();
    assert.equal(detects,0);
    await page.locator('#auto-detect-boxes').click();
    await page.waitForFunction(()=>document.querySelectorAll('.reference-selected-thumbnail').length===10);
    assert.equal(detects,1);
    assert.equal(await page.locator('.reference-proposal-marker').count(),30,'all forty suggestions stay available after ten are selected');
    await page.locator('#reference-select-areas').click();
    await page.locator('.reference-selected-thumbnail').first().click();
    await page.keyboard.press('x');
    assert.equal(await page.locator('.reference-selected-thumbnail').count(),9);
    await page.locator('#reference-image-viewport').focus();await page.keyboard.press('Space');
    assert.equal(await page.locator('.reference-selected-thumbnail').count(),10);
    assert.equal(await page.locator('.coordinate-entry').isVisible(),false);
    for(const width of [1440,390,320]) {
      await page.setViewportSize({width,height:1100});await page.locator('#reference-editor').scrollIntoViewIfNeeded();
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),width);
      await noJargon();await page.screenshot({path:path.join(out,'examples-'+width+'.png'),fullPage:true});
    }
    await page.setViewportSize({width:1440,height:1100});
    await page.locator('#reference-zoom-in').click();
    assert.equal(await page.locator('#reference-zoom-reset').innerText(),'150%');
    await page.locator('#reference-zoom-reset').click();
    await page.locator('#reference-image-viewport').scrollIntoViewIfNeeded();
    bounds=await page.locator('#box-overlay').boundingBox();
    await page.mouse.move(bounds.x+bounds.width*.4,bounds.y+bounds.height*.4);
    assert.equal(await page.locator('#reference-magnifier').isVisible(),true);
    await page.keyboard.down('Alt');await page.mouse.wheel(0,-120);await page.keyboard.up('Alt');
    assert.equal(await page.locator('#reference-zoom-reset').innerText(),'100%');
    await page.locator('#start-button').click();
    await page.waitForFunction(()=>document.querySelector('#metric-total').textContent==='14');
    assert.equal(submitted.model,'countgd-plusplus');assert.equal(submitted.confidence,'0.8');
    assert.equal(submitted.boxes[0].length,10);assert.equal(submitted.label,'');
    await noJargon();await page.screenshot({path:path.join(out,'results-desktop.png'),fullPage:true});
    await page.getByRole('button',{name:'Review photo',exact:true}).click();
    await page.locator('#review-image').waitFor();
    await noJargon();assert.equal(await page.locator('#review-count').innerText(),'1 counted');
    await page.locator('#review-save').click();
    await page.waitForFunction(()=>!document.querySelector('#review-confirm').disabled);
    await page.locator('#review-confirm').click();
    await page.locator('#review-dialog').waitFor({state:'hidden'});
    assert.equal(await page.locator('#review-dialog').isVisible(),false);
    // A later quick batch needs areas, but no selected examples or suggestion request.
    await page.locator('#next-gallery').setInputFiles(photo);
    await page.locator('#editor-image').waitFor();
    await page.waitForFunction(()=>!document.querySelector('#quick-upload-button').disabled);
    assert.equal(await page.locator('.reference-selected-thumbnail').count(),0);
    await page.locator('#quick-upload-button').click();
    await page.waitForFunction(()=>document.querySelector('#notice').textContent.includes('Mark an area'));
    await page.locator('#box-overlay').scrollIntoViewIfNeeded();
    bounds=await page.locator('#box-overlay').boundingBox();
    await page.mouse.move(bounds.x+bounds.width*.1,bounds.y+bounds.height*.1);await page.mouse.down();
    await page.mouse.move(bounds.x+bounds.width*.7,bounds.y+bounds.height*.7,{steps:8});await page.mouse.up();
    const marked=await page.locator('.reference-detection-area').evaluateAll(items=>items.map(item=>JSON.parse(item.dataset.box)));
    assert.equal(marked.length,1);
    for(const width of [1440,390,320]) {
      await page.setViewportSize({width,height:1100});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),width);
      assert.equal(await page.locator('#quick-upload-button').isVisible(),true);
    }
    const before=detects;
    await page.locator('#quick-upload-button').click();
    await page.waitForFunction(()=>document.querySelectorAll('#batch-list .batch-card').length===2);
    assert.equal(detects,before,'quick counting is one batch request; no reference suggestions');
    assert.equal(uploads.at(-1).countMode,'quick');
    assert.deepEqual(uploads.at(-1).searchAreas,[marked]);
    assert.deepEqual(uploads.at(-1).boxes,[]);
    assert.match(await page.locator('#batch-list').innerText(),/Quick count/);
    await page.reload();
    await page.waitForFunction(()=>document.querySelectorAll('#batch-list .batch-card').length===2);
    assert.match(await page.locator('#batch-list').innerText(),/Quick count/);
    await noJargon();
    await page.screenshot({path:path.join(out,'quick-results-mobile.png'),fullPage:true});

    // Starting a fresh quick count also bypasses examples and stays alongside the main action.
    await page.evaluate(()=>localStorage.removeItem('countwell.sessionId'));
    await page.reload();
    await page.waitForFunction(()=>!document.querySelector('#quick-count-button').disabled);
    await page.locator('#target-gallery').setInputFiles(photo);
    await page.locator('#editor-image').waitFor();
    await page.locator('#quick-count-button').click();
    await page.waitForFunction(()=>document.querySelector('#notice').textContent.includes('Mark an area'));
    await page.locator('#box-overlay').scrollIntoViewIfNeeded();bounds=await page.locator('#box-overlay').boundingBox();
    await page.mouse.move(bounds.x+.001,bounds.y+.001);await page.mouse.down();
    await page.mouse.move(bounds.x+bounds.width-.001,bounds.y+bounds.height-.001,{steps:8});await page.mouse.up();
    await page.setViewportSize({width:1440,height:1100});
    await page.screenshot({path:path.join(out,'quick-start-desktop.png'),fullPage:true});
    await page.locator('#quick-count-button').click();
    await page.waitForFunction(()=>document.querySelector('#metric-total').textContent==='3');
    assert.equal(uploads.at(-1).countMode,'quick');
    assert.equal(uploads.at(-1).boxes.length,0);
    assert.equal(uploads.at(-1).searchAreas.length,1);
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({selected:10,detects,uploads,errors,widths:[1440,390,320]},null,2));
    console.log('PASS: standard and Quick count, required marked areas, no examples for Quick count, later batches, reload, desktop/mobile, review and shortcuts');
  }finally{await browser.close();await new Promise(resolve=>server.close(resolve));
    const resolved=fs.realpathSync(temp);assert.equal(path.dirname(resolved),fs.realpathSync(os.tmpdir()));
    assert.ok(path.basename(resolved).startsWith('simple-counting-e2e-'));fs.rmSync(resolved,{recursive:true,force:true});}
})().catch(e=>{console.error(e);process.exitCode=1;});
