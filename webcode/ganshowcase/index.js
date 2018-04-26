import "babel-polyfill";
import * as tf from '@tensorflow/tfjs';


// --------------------------------------------------------------------------------------------------------
// Configuration
// --------------------------------------------------------------------------------------------------------


let all_model_info = {
    dcgan64: {
        description: 'DCGAN, 64x64 (16 MB)',
        model_url: "https://storage.googleapis.com/store.alantian.net/tfjs_gan/chainer-dcgan-celebahq-64/tfjs_SmoothedGenerator_50000/model.json",
        model_size: 64,
        model_latent_dim: 128,
        draw_multiplier: 4,
        animate_frame: 200,
    },
    resnet128: {
        description: 'ResNet, 128x128 (252 MB)',
        model_url: "https://storage.googleapis.com/store.alantian.net/tfjs_gan/chainer-resent128-celebahq-128/tfjs_SmoothedGenerator_20000/model.json",
        model_size: 128,
        model_latent_dim: 128,
        draw_multiplier: 2,
        animate_frame: 10
    },
    resnet256: {
        description: 'ResNet, 256x256 (252 MB)',
        model_url: "https://storage.googleapis.com/store.alantian.net/tfjs_gan/chainer-resent256-celebahq-256/tfjs_SmoothedGenerator_40000/model.json",
        model_size: 256,
        model_latent_dim: 128,
        draw_multiplier: 1,
        animate_frame: 10
    }
};

let default_model_name = 'dcgan64';


// --------------------------------------------------------------------------------------------------------
// Computing
// --------------------------------------------------------------------------------------------------------

function computing_prep_canvas(size) {
    // We don't `return tf.image.resizeBilinear(v1, [size * draw_multiplier, size * draw_multiplier]);`
    // since that makes image blurred, which is not what we want.
    // So instead, we manually enlarge the image.
    let canvas = document.getElementById("the_canvas");
    let ctx = canvas.getContext("2d");
    ctx.canvas.width = size;
    ctx.canvas.height = size;
}

function image_enlarge(y, draw_multiplier) {
    if (draw_multiplier === 1) {
        return y;
    }
    let size = y.shape[0];
    return y.expandDims(2).tile([1, 1, draw_multiplier, 1]
    ).reshape([size, size * draw_multiplier, 3]
    ).expandDims(1).tile([1, draw_multiplier, 1, 1]
    ).reshape([size * draw_multiplier, size * draw_multiplier, 3])
}

async function computing_animate_latent_space(model, draw_multiplier, animate_frame) {
    const inputShape = model.inputs[0].shape.slice(1);
    const shift = tf.randomNormal(inputShape).expandDims(0);
    const freq = tf.randomNormal(inputShape, 0, .1).expandDims(0);

    let c = document.getElementById("the_canvas");
    let i = 0;
    while (i < animate_frame) {
        i++;
        const y = tf.tidy(() => {
            const z = tf.sin(tf.scalar(i).mul(freq).add(shift));
            const y = model.predict(z).squeeze().transpose([1, 2, 0]).div(tf.scalar(2)).add(tf.scalar(.5));
            return image_enlarge(y, draw_multiplier);
        });

        await tf.toPixels(y, c);
        await tf.nextFrame();
    }
}

async function computing_generate_main(model, size, draw_multiplier, latent_dim) {
    const y = tf.tidy(() => {
        const z = tf.randomNormal([1, latent_dim]);
        const y = model.predict(z).squeeze().transpose([1, 2, 0]).div(tf.scalar(2)).add(tf.scalar(0.5));
        return image_enlarge(y, draw_multiplier);

    });
    let c = document.getElementById("the_canvas");
    await tf.toPixels(y, c);
}

const ui_delay_before_tf_computing_ms = 2000;  // Delay that many ms before tf computing, which can block UI drawing.

function resolve_after_ms(x, ms) {
    return new Promise(resolve => {
        setTimeout(() => {
            resolve(x);
        }, ms);
    });
}

class ModelRunner {
    constructor() {
        this.model_promise_cache = {};
        this.model_promise = null;
        this.model_name = null;
        this.start_time = null;
    }

    setup_model(model_name) {
        this.model_name = model_name;
        let model_info = all_model_info[model_name];
        let model_size = model_info.model_size,
            model_url = model_info.model_url,
            draw_multiplier = model_info.draw_multiplier,
            description = model_info.description;

        computing_prep_canvas(model_size * draw_multiplier);
        ui_set_canvas_wrapper_size(model_size * draw_multiplier);
        ui_logging_set_text(`Setting up model ${description}...`);

        if (model_name in this.model_promise_cache) {
            this.model_promise = this.model_promise_cache[model_name];
            ui_logging_set_text(`Model "${description}" is ready.`);
        } else {
            ui_generate_button_disable('Loading...');
            ui_animate_button_disable('Loading...');
            ui_logging_set_text(`Loading model "${description}"...`);
            this.model_promise = tf.loadModel(model_url);
            this.model_promise.then((model) => {
                return resolve_after_ms(model, ui_delay_before_tf_computing_ms);
            }).then((model) => {
                ui_generate_button_enable();
                ui_animate_button_enable();
                ui_logging_set_text(`Model "${description}" is ready.`);
            });
            this.model_promise_cache[model_name] = this.model_promise;
        }
    }

    generate() {
        let model_info = all_model_info[this.model_name];
        let model_size = model_info.model_size,
            model_latent_dim = model_info.model_latent_dim,
            draw_multiplier = model_info.draw_multiplier;

        computing_prep_canvas(model_size * draw_multiplier);

        ui_logging_set_text('Generating image...');
        ui_generate_button_disable('Generating...');
        ui_animate_button_disable();

        this.model_promise.then((model) => {
            return resolve_after_ms(model, ui_delay_before_tf_computing_ms);
        }).then((model) => {
            this.start_ms = (new Date()).getTime();
            return computing_generate_main(model, model_size, draw_multiplier, model_latent_dim);
        }).then((_) => {
            let end_ms = (new Date()).getTime();
            ui_generate_button_enable();
            ui_animate_button_enable();
            ui_logging_set_text(`Image generated. It took ${(end_ms - this.start_ms)} ms.`);
        });
    }

    animate() {
        let model_info = all_model_info[this.model_name];
        let model_size = model_info.model_size,
            model_latent_dim = model_info.model_latent_dim,
            draw_multiplier = model_info.draw_multiplier,
            animate_frame = model_info.animate_frame;

        computing_prep_canvas(model_size * draw_multiplier);

        ui_logging_set_text('Animating latent space...');
        ui_generate_button_disable();
        ui_animate_button_disable('Animating...');

        this.model_promise.then((model) => {
            return resolve_after_ms(model, ui_delay_before_tf_computing_ms);
        }).then((model) => {
            this.start_ms = (new Date()).getTime();
            return computing_animate_latent_space(model, draw_multiplier, animate_frame);
        }).then((_) => {
            let end_ms = (new Date()).getTime();
            ui_generate_button_enable();
            ui_animate_button_enable();
            ui_logging_set_text(`Animating took ${(end_ms - this.start_ms)} ms.`);
        });
    }
}

let model_runner = new ModelRunner();


// --------------------------------------------------------------------------------------------------------
// Controlling / UI
// --------------------------------------------------------------------------------------------------------

function change_model(model_name) {
    model_runner.setup_model(model_name);
}

function ui_set_canvas_wrapper_size(size) {
    document.getElementById('canvas-wrapper').style.height = size.toString() + "px";
    document.getElementById('canvas-wrapper').style.width = size.toString() + "px";
}

const generate_button_default_text = "Generate";

function ui_generate_button_disable(text) {
    document.getElementById('generate-button').classList.add("disabled");
    text = (text === undefined) ? generate_button_default_text : text;
    document.getElementById('generate-button').textContent = text;
}

function ui_generate_button_enable() {
    document.getElementById('generate-button').classList.remove("disabled");
    document.getElementById('generate-button').textContent = generate_button_default_text;
}

const animate_button_default_text = "Animate";

function ui_animate_button_disable(text) {
    document.getElementById('animate-button').classList.add("disabled");
    text = (text === undefined) ? animate_button_default_text : text;
    document.getElementById('animate-button').textContent = text;
}

function ui_animate_button_enable() {
    document.getElementById('animate-button').classList.remove("disabled");
    document.getElementById('animate-button').textContent = animate_button_default_text;
}

function ui_logging_set_text(text) {
    text = (text === undefined) ? generate_button_default_text : text;
    document.getElementById('logging').textContent = text;
}

function ui_generate_button_event_listener(event) {
    model_runner.generate();
}

function ui_animate_button_event_listener(event) {
    model_runner.animate();
}

function ui_change_model_event_listener(event) {
    let value = event.target.value;
    change_model(value);
}

function ui_setup_model_select() {
    let model_select_elem = document.getElementById('model-select');
    for (let model_name in all_model_info) {
        let model_info = all_model_info[model_name];

        let option_node = document.createElement('option');
        option_node.setAttribute('value', model_name);
        option_node.textContent = model_info.description;

        if (model_name === default_model_name) {
            option_node.selected = true;
        }
        model_select_elem.appendChild(option_node);
    }

    let instance = M.FormSelect.init(model_select_elem, {});
    model_select_elem.onchange = ui_change_model_event_listener;

}

function ui_setup_generate_button() {
    ui_generate_button_enable();
    document.getElementById('generate-button').onclick = ui_generate_button_event_listener;
}


function ui_setup_animate_button() {
    ui_animate_button_enable();
    document.getElementById('animate-button').onclick = ui_animate_button_event_listener;
}

function ui_setup() {
    ui_setup_model_select();

    ui_setup_generate_button();
    ui_setup_animate_button();

    change_model(default_model_name);
}


// --------------------------------------------------------------------------------------------------------

function main() {
    ui_setup();
};


main();
