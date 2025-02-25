const fs = require("fs");
const tfnode = require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");

const dirName = "anime1";

const numImg = 1000;
const seed_size = 100;
const batchSize = 64;
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
    return imgContent;
}


function getBatch(items) {
    return tf.data.array(items).batch(batchSize);
}

async function normalize() {
  const data = getImage();
  //const arr = await data.toArray();
  let dataArr = [];
  for (let i=0; i<numImg; i++) {
      let batch = data[i];
      batch = tf.cast(batch, dtype = 'float32');
      batch = (batch.sub(tf.scalar(127.5))).div(tf.scalar(127.5));
      dataArr.push(batch);
  }
  return dataArr;
}

async function getDataset() {
  const dataset = await normalize();
  const data = getBatch(dataset);
  return data;
}

const data = getDataset();

const init = tf.initializers.truncatedNormal(3);
init.stddev = 0.02;
init.DEFAULT_STDDEV = 0.02;

const combined_optimizer = tf.train.adam(0.0002,0.5);
const generator_optimizer = tf.train.adam(0.0002,0.5);
const discriminator_optimizer = tf.train.adam(0.0002,0.5);

function build_generator(seed_size) {
    const model = tf.sequential();
    const epsilon = 0.00001;

    // Block 1
    model.add(tf.layers.dense({
        units : 4*4*512,
        inputShape : [seed_size],
        activation : 'linear'
    }));
    model.add(tf.layers.leakyReLU({alpha : 0.2}));
    model.add(tf.layers.reshape({
        targetShape : [4, 4, 512]
    }));

    // Block 2
    model.add(tf.layers.conv2dTranspose({
      filters : 512,
      kernelSize : 4,
      strides : 2,
      padding : 'same',
      kernelInitializer : init,
    }));
    model.add(tf.layers.batchNormalization({
      epsilon : epsilon,
      momentum : 0.9
    }));
    model.add(tf.layers.leakyReLU({alpha : 0.2}));

    // Block 3
    model.add(tf.layers.conv2dTranspose({
      filters : 256,
      kernelSize : 4,
      strides : 2,
      padding : 'same',
      kernelInitializer : init,
    }));
    model.add(tf.layers.batchNormalization({
      epsilon : epsilon,
      momentum : 0.9
    }));
    model.add(tf.layers.leakyReLU({alpha : 0.2}));

    // Block 4
    model.add(tf.layers.conv2dTranspose({
      filters : 128,
      kernelSize : 4,
      strides : 2,
      padding : 'same',
      kernelInitializer : init,
    }));
    model.add(tf.layers.batchNormalization({
      epsilon : epsilon,
      momentum : 0.9
    }));
    model.add(tf.layers.leakyReLU({alpha : 0.2}));

    // Block 5
    model.add(tf.layers.conv2dTranspose({
      filters : 64,
      kernelSize : 4,
      strides : 2,
      padding : 'same',
      kernelInitializer : init,
    }));
    model.add(tf.layers.batchNormalization({
      epsilon : epsilon,
      momentum : 0.9
    }));
    model.add(tf.layers.leakyReLU({alpha : 0.2}));

    // Block 6
    model.add(tf.layers.conv2dTranspose({
      filters : 3,
      kernelSize : 4,
      strides : 1,
      padding : 'same',
      kernelInitializer : init,
    }));

    model.add(tf.layers.activation({
      activation : 'tanh'
    }));

    const inp = tf.layers.input({shape : [seed_size]});
    const op = model.apply(inp);

    return tf.model({inputs : inp, outputs: op});
}

function build_discriminator(image_length, image_channels) {
    const model = tf.sequential();

    // Block 1
    model.add(tf.layers.conv2d({
        filters : 128,
        kernelSize : 3,
        padding : "same",
        inputShape : [image_length, image_length, image_channels],
    }));
    model.add(tf.layers.leakyReLU({alpha : 0.2}));
    model.add(tf.layers.batchNormalization());

    model.add(tf.layers.conv2d({
      filters : 128,
      kernelSize : 3,
      padding : "same",
    }));
    model.add(tf.layers.leakyReLU({alpha : 0.2}));
    model.add(tf.layers.batchNormalization());
    
    model.add(tf.layers.maxPool2d({
      poolSize : [3, 3]
    }));
    model.add(tf.layers.dropout({rate : 0.2}));

    // Block 2
    model.add(tf.layers.conv2d({
      filters : 128,
      kernelSize : 3,
      padding : "same",
      inputShape : [image_length, image_length, image_channels],
    }));
    model.add(tf.layers.leakyReLU({alpha : 0.2}));
    model.add(tf.layers.batchNormalization());

    model.add(tf.layers.conv2d({
      filters : 128,
      kernelSize : 3,
      padding : "same",
    }));
    model.add(tf.layers.leakyReLU({alpha : 0.2}));
    model.add(tf.layers.batchNormalization());
    
    model.add(tf.layers.maxPool2d({
      poolSize : [3, 3]
    }));
    model.add(tf.layers.dropout({rate : 0.3}));

    // Block 3
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units : 128}));
    model.add(tf.layers.leakyReLU({alpha : 0.2}));
    model.add(tf.layers.dense({units : 128}));
    model.add(tf.layers.leakyReLU({alpha : 0.2}));
    model.add(tf.layers.dense({
      units : 1,
      activation : 'sigmoid'
    }));

    const inp = tf.layers.input({shape : [image_length, image_length, image_channels]});
    const op = model.apply(inp);

    return tf.model({inputs : inp, outputs: op});
    
}

const discriminator = build_discriminator(64, 3);
discriminator.compile({optimizer : discriminator_optimizer, loss : 'binaryCrossentropy', metrics : ['accuracy']});

const generator = build_generator(seed_size);
generator.compile({optimizer : generator_optimizer, loss : 'binaryCrossentropy'});

function gan_model() {
  discriminator.trainable = false;
  const model = tf.sequential();
  model.add(generator);
  model.add(discriminator);
  model.compile({optimizer : combined_optimizer, loss : 'binaryCrossentropy'});
  return model;
}
const combined = gan_model();

async function train(epochs) {
    for (let i=0; i<epochs; i++){
        let gloss = 0;
        let dloss = 0;
        let accuracy = 0;
        const dataArr = await (await data).toArray();
        for (let j=0; j<numBatch; j++) {

            const batch = dataArr[j];
            const batch_size = batch.shape[0];
            const noise = tf.randomNormal([batch_size, seed_size], 0, 1);
            const generated_image = generator.predict(noise);

            const disc_loss_real = await discriminator.trainOnBatch(batch, tf.ones([batch_size,1]));
            const disc_loss_fake = await discriminator.trainOnBatch(generated_image, tf.zeros([batch_size, 1]));  
            const dlossBatch = 0.5*(disc_loss_fake[0] + disc_loss_real[0]);

            const accBatch = 0.5*(disc_loss_fake[1] + disc_loss_real[1]);
            const glossBatch = await combined.trainOnBatch(noise, tf.ones([batch_size]));

            gloss = gloss + glossBatch;
            dloss = dloss + dlossBatch;
            accuracy = accuracy + accBatch;
        };

        const mean_gloss = gloss / numBatch;
        const mean_dloss = dloss / numBatch;
        const mean_accuracy = 100 * (accuracy / numBatch);

        console.log("Epoch: " + (i+1));
        console.log("Generator loss: "+mean_gloss);
        console.log("Disciminator loss: "+mean_dloss);
        console.log("Accuracy: "+mean_accuracy+"%"+"\n");
    }
    await generator.save("file://./my-model-1");
    
}


train(40);

//generator.summary();

//discriminator.summary();

