// --------------------------------------------------------------------------------------------------------
// Configuration
// --------------------------------------------------------------------------------------------------------


let all_model_info = {
    resnet256_wasm: {
        description: 'ResNet, WASM, 256x256, 8bit',
        // model_url: "https://srv0.alantian.net/public//share/tfjs_gan/chainer-resent256-celebahq-256/tfjs_SmoothedGenerator_40000/model.json",
        model_url: '/resnet256_8bit_wasm',
        model_size: 256,
        model_latent_dim: 128,
        draw_multiplier: 1,
    },
    resnet256_webgpu: {
        description: 'ResNet, WebGPU, 256x256, 8bit',
        // model_url: "https://srv0.alantian.net/public//share/tfjs_gan/chainer-resent256-celebahq-256/tfjs_SmoothedGenerator_40000/model.json",
        model_url: '/resnet256_8bit_webgpu',
        model_size: 256,
        model_latent_dim: 128,
        draw_multiplier: 1,
    }
};

let default_model_name = 'resnet256_wasm';


// --------------------------------------------------------------------------------------------------------
// Computing
// --------------------------------------------------------------------------------------------------------



function computing_prep_canvas(size) {
    let canvas = document.getElementById("the_canvas");
    let ctx = canvas.getContext("2d");
    ctx.canvas.width = size;
    ctx.canvas.height = size;
}

function uniformToNormal(u, v) {
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function randomNormal() {
    var u = 1 - Math.random();
    var v = 1 - Math.random();
    return uniformToNormal(u, v);
}

function randomNormalArray(size) {
    var res = new Array(size)
    for (var i = 0; i<size; i++) {
      res[i] = randomNormal()
    }
    return res
}

async function computing_sample(num, latent_dim, model) {
    let z = randomNormalArray(latent_dim) //tf.randomNormal([num, latent_dim]);
    model.inputs[0].set(z)
    await model.run()
    let output = model.outputs[0].toActual();// .toActual();
    console.log(output)
    return output
    // let res = runner.outputs[0]
}

async function computing_generate_main(model, size, draw_multiplier, latent_dim) {
    var generator_result;
    generator_result = await computing_sample(1, latent_dim, model);
    console.log(generator_result)
    WebDNN.Image.setImageArrayToCanvas(generator_result, 256, 256, document.getElementById('the_canvas'),
    {
        scale: [127.5, 127.5, 127.5],
        bias: [127.5, 127.5, 127.5],
        order: WebDNN.Image.Order.CHW,
        dstW: 256,
        dstH: 256
    })
}


function computing_prep_canvas(size) {
    let canvas = document.getElementById("the_canvas");
    let ctx = canvas.getContext("2d");
    ctx.canvas.width = size;
    ctx.canvas.height = size;
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
    }

    setup_model(model_name) {
        this.model_name = model_name;
        let model_info = all_model_info[model_name];
        let model_size = model_info.model_size,
            model_url = model_info.model_url,
            draw_multiplier = model_info.draw_multiplier,
            description= model_info.description;

        computing_prep_canvas(model_size * draw_multiplier);
        ui_set_canvas_wrapper_size(model_size * draw_multiplier);
        ui_logging_set_text(`Setting up model ${description}...`);

        if (model_name in this.model_promise_cache) {
            this.model_promise = this.model_promise_cache[model_name];
            ui_logging_set_text(`Model "${description}" is ready.`);
        } else {
            ui_generate_button_disable('Loading...');
            ui_logging_set_text(`Loading model "${description}"...`);
            this.model_promise = WebDNN.load(model_url);
            this.model_promise.then((model) => {
                return resolve_after_ms(model, ui_delay_before_tf_computing_ms);
            }).then((model) => {
                ui_generate_button_enable();
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

        ui_generate_button_disable('Generating...');
        ui_logging_set_text('Generating image...');
        this.model_promise.then((model) => {
            return resolve_after_ms(model, ui_delay_before_tf_computing_ms);
        }).then( async (model) => {
            let start_ms = (new Date()).getTime();
            await computing_generate_main(model, model_size, draw_multiplier, model_latent_dim)
            let end_ms = (new Date()).getTime();
            ui_generate_button_enable();
            ui_logging_set_text(`Image generated. It took ${(end_ms - start_ms)} ms.`);
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

function ui_logging_set_text(text){
    text = (text === undefined) ? generate_button_default_text : text;
    document.getElementById('logging').textContent = text;
}

function ui_generate_button_event_listener(event) {
    model_runner.generate();
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

function ui_setup() {
    ui_setup_model_select();

    ui_setup_generate_button();

    change_model(default_model_name);
}


// --------------------------------------------------------------------------------------------------------

function main() {
    ui_setup();
};


main();
