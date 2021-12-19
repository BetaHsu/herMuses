// AVAILABLE MODELS TO TRY
// More detailed style representation, large file
const STYLE_BIG_MODEL =
  "https://reiinakano.com/arbitrary-image-stylization-tfjs/saved_model_style_inception_js/model.json";
// More accurate transform, less artistic interpretation, large file
const TRANSFORMER_BIG_MODEL =
  "https://reiinakano.com/arbitrary-image-stylization-tfjs/saved_model_transformer_js/model.json";
// Less detailed style representation, more noise, smaller file
const STYLE_SMALL_MODEL =
  "https://reiinakano.com/arbitrary-image-stylization-tfjs/saved_model_style_js/model.json";
// Less accurate transform, more artistic interpretation, small file, more like painting cuz it's filling in the empty spaces
const TRANSFORMER_SMALL_MODEL =
  "https://reiinakano.com/arbitrary-image-stylization-tfjs/saved_model_transformer_separable_js/model.json";

// SETTINGS
const STYLE_RATIO = 0.85;
const MAX_STYLE_SIZE = 400;
const MAX_FACE_SIZE = 400;
const STYLE_MODEL = STYLE_SMALL_MODEL;
const TRANSFORMER_MODEL = TRANSFORMER_SMALL_MODEL;

// * START- Special thanks to the help from Chase C. for TensorFlow and P5.js utilities*/
//(Code understood, commented, and further modified by me 🙈)

// cacheFetch so when an asset is requested, it is only fetched once
let cache;
async function cacheFetch(url, opts) {
  cache =
    cache ??
    (await ("caches" in self
      ? caches.open("style-transfer-nn")
      : Promise.resolve(null)));

  if (cache != null) {
    const maybeResponse = await cache.match(url);
    if (maybeResponse != null) return maybeResponse;

    const res = await fetch(url, opts);
    cache.put(url, res.clone());
    return res;
  }
  return fetch(url, opts);
}

async function loadTFGraphModel(url) {
  return tf.loadGraphModel(url, { fetchFunc: cacheFetch });
}

async function asyncLoadImage(url) {
  // fetch the url to make a "html" image, not P5Image, cuz tensorflow use html image
  const res = await cacheFetch(url);
  const blob = await res.blob();
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      resolve(img);
    };
    img.src = URL.createObjectURL(blob);
  });
}

async function asyncP5LoadImage(url) {
  // fetch the url to make a "P5image", for drawing directly on the canvas
  const res = await cacheFetch(url);
  const blob = await res.blob();
  return new Promise((resolve) => {
    loadImage(URL.createObjectURL(blob), (img) => {
      resolve(img);
    });
  });
}

// take whatever image(style or face), check if the largest dimentions is larger than maxSize,
// then squeeze the largest dimention of the image to maxSize and squeeze the other dimention proportionally smaller
function maxSizeImage(img, max_size) {
  if (img.height > img.width) {
    const newHeight = min(max_size, img.height);
    const newWidth = (img.width * newHeight) / img.height; //modifying it Proportionally smaller
    img.height = newHeight;
    img.width = newWidth;
  } else {
    const newWidth = min(max_size, img.width);
    const newHeight = (img.height * newWidth) / img.width; //modifying it Proportionally smaller
    img.height = newHeight;
    img.width = newWidth;
  }
}
//reference: https://github.com/reiinakano/arbitrary-image-stylization-tfjs/blob/master/main.js
async function generateStyleRepresentation(
  styleNet,
  faceImg,
  styleImg,
  styleRatio = 1.0,
  maxStyleSize = 250
) {
  // if style ratio = 1, then don't need faceImage Style
  maxSizeImage(styleImg, maxStyleSize);
  let bottleneck = await tf.tidy(() => {
    return styleNet.predict(
      //tf.browser.fromPixels creates a tf.Tensor from an image,
      // in shape of [r, g, b, r, g, b, r, g, b, ...] as integers from 0 to 255
      // toFloat() casts the array to float type : 0.0 ~ 225.0
      //.div(tf.scalar(255)) divides all by a scalar (1 dimensional number) 255, makes all into value 0.0 ~ 1.0
      //.expandDims() inserts a dimension into the tensor's shape
      //(ex: x = tf.tensor1d([1, 0, 0.5]); x.expandDims(axis) thus makes x to [[1], [0], [0.5]])
      tf.browser.fromPixels(styleImg).toFloat().div(tf.scalar(255)).expandDims()
    );
  });
  // if style ratio < 1, then styleImage Style + faceImage Style
  if (styleRatio < 1.0) {
    const identityBottleneck = await tf.tidy(() => {
      //.tidy clean up when the tidy ends
      return styleNet.predict(
        //.predict Generates output predictions for the input samples.
        tf.browser
          .fromPixels(faceImg)
          .toFloat()
          .div(tf.scalar(255)) //Divides two tf.Tensors element-wise, A / B.
          .expandDims() //tf.expandDims (x, axis?) Returns a tf.Tensor that has expanded rank
      );
    });
    const styleBottleneck = bottleneck;
    bottleneck = await tf.tidy(() => {
      //t.scalar( value, dataType ), scaler is a zero-dimension array, called as a rank-0 Tensor //mul multiply
      const styleBottleneckScaled = styleBottleneck.mul(tf.scalar(styleRatio));
      const identityBottleneckScaled = identityBottleneck.mul(
        tf.scalar(1.0 - styleRatio)
      );
      //.addStrict adds two tf.Tensors element-wise, A + B.
      return styleBottleneckScaled.addStrict(identityBottleneckScaled);
    });
    //.dispose throws away result when running model's prediction, for more memory capacity, dispose tf.Tensors found within the provided object,
    styleBottleneck.dispose();
    identityBottleneck.dispose();
  }
  return bottleneck;
}

async function stylizeImage(transformNet, styleRep, faceImg, maxFaceSize) {
  // use the transforNet to take the styleRepresentation and apply it on faceImage
  maxSizeImage(faceImg, maxFaceSize);
  return tf.tidy(() => {
    return transformNet
      .predict([
        tf.browser
          .fromPixels(faceImg)
          .toFloat()
          .div(tf.scalar(255))
          .expandDims(),
        styleRep
      ])
      .squeeze();
  });
}

async function tfImageRepToP5Image(tfImageRep, width, height) {
  const img = createImage(width, height);
  //turns representation to pixels and back to P5img (but in fact is p5's canvas, cuz in p5 it's not an actual image)
  await tf.browser.toPixels(tfImageRep, img.canvas); //tf.browser.toPixels(img, Canvas);
  return img;
}

// for memoize
function fixNumberKey(maybeNumber) {
  if (typeof maybeNumber === "number") {
    return maybeNumber.toFixed(6);
  } else {
    return maybeNumber;
  }
}
//store the result for funrction (ex: generateStylizedP5Image()), cuz we run it repetitively, but too computational intensive
function memoize(func) {
  const memoized = function (...args) {
    let cache = memoized.cache;
    for (let i = 0; i < args.length - 1; i++) {
      let cacheLevel = cache.get(fixNumberKey(args[i]));
      if (cacheLevel == null) {
        cacheLevel = new Map();
        cache.set(args[i], cacheLevel);
      }
      cache = cacheLevel;
    }
    const lastArg = fixNumberKey(args[args.length - 1]);
    if (cache.has(lastArg)) {
      return cache.get(lastArg);
    }

    const result = func.apply(this, args);
    cache.set(lastArg, result);
    return result;
  };
  memoized.cache = new Map();
  return memoized;
}
// memoize the repeted actions once it's run once it remember the actions
const generateStylizedP5Image = memoize(async function generateStylizedP5Image(
  styleNet,
  transformNet,
  faceImg,
  styleImg,
  styleRatio
) {
  await tf.nextFrame();
  // representation: return a result of "how it represente" the style, not an image, Interpreted by the neural network,
  const styleRep = await generateStyleRepresentation(
    //make the representation for the neuralnetwork, for the given style image
    styleNet, // style neural network
    faceImg, //need faceImg cuz if styleRatio !=1, the styleNet still always need to geenrate representation partially according to original image (see definition)
    styleImg,
    styleRatio,
    MAX_STYLE_SIZE
  );

  await tf.nextFrame();
  // stylized: a representation(merely numbers) of an image
  const stylized = await stylizeImage(
    //take the representation, transfor a faceImage into a styled image
    transformNet, // transfer neural network: transfer a style to an image
    styleRep, //the representation of the styled neural network
    faceImg,
    MAX_FACE_SIZE
  );

  // takes representation of an image, turns it into pixels, and applies to a new p5 image
  const result = await tfImageRepToP5Image(
    stylized,
    faceImg.width, // match with the face image dimensions
    faceImg.height
  );
  styleRep.dispose(); //tf.dispose
  stylized.dispose();
  return result;
});

/* END: TensorFlow and P5.js utilities */

function copyRectangleFromP5CanvasToHTMLImageData(x, y, width, height) {
  return drawingContext.getImageData(x, y, width, height);
}
