const fs = require("fs");
const tfnode = require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");

const dirName = "anime1";

const numImg = 201;
const batchSize = 100;
let numBatch;
if (numImg % batchSize == 0) {
  numBatch = numImg / batchSize;
} else {
  numBatch = Math.floor(numImg / batchSize) + 1;
}

const imgPaths = [];
for (let i=0; i<numImg; i++) {
    imgPaths.push(dirName.concat("/".concat((i+1).toString(), ".png")));
}

function getImage() {
    const imgContent = [];

    for (let i=0; i<imgPaths.length; i++) {
        var content = fs.readFileSync(imgPaths[i]);
        var tfImg = tfnode.node.decodeImage(content, 3);
        imgContent.push(tfImg);
        
    }
    /*
    for (let i=0; i<imgContent.length; i++) {
        console.log(imgContent[i]);
    }
     */
    return imgContent;
}


function getBatch(items) {
    return tf.data.array(items).batch(batchSize);
    //return tf.data.array(items);
}

/*
const tfImg = getBatch(getImage(), 5);
tfImg.forEachAsync(imgArr => console.log(imgArr));
*/


async function getDataset() {
    const data = getImage();
    //const arr = await data.toArray();
    let dataArr = [];
    for (let i=0; i<numImg; i++) {
        let batch = data[i];
        batch = tf.cast(batch, dtype = 'float32');
        batch = batch.sub(tf.scalar(127.5)).div(tf.scalar(127.5));
        dataArr.push(batch);
    }
    return dataArr;
}


/*
function getRandomSubarray(arr, size) {
    var shuffled = arr.slice(0), i = arr.length, temp, index;
    while (i--) {
        index = Math.floor((i + 1) * Math.random());
        temp = shuffled[index];
        shuffled[index] = shuffled[i];
        shuffled[i] = temp;
    }
    return shuffled.slice(0, size);
}
*/
/*
function getDataset() {
    return getBatch(getImage(), batch_size);
}
*/
async function printArray() {
    const dataset = await getDataset();
    //for (let i=0; i<1; i++) {
        //const i = 0;
        //console.log(data[i]);
        //data[i].print();    
        
    //}
    //const dataArr = data.toArray();
    //const batch = (await data)[2];
    const data = getBatch(dataset);
    const dataArr = data.toArray();
    const batch = (await dataArr)[2];
    console.log(batch.shape[0]);
    //batch.print();
    
}
//printArray();
const seed_size = 100;
const noise = tf.randomNormal([1,seed_size]);
noise.print();
