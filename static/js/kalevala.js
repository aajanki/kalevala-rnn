import * as tf from '@tensorflow/tfjs';

class TextSampler {
    constructor(model, char2idx) {
        this.model = model;
        this.char2idx = char2idx;
        this.idx2char = this.reverseChar2idx(char2idx);
    }

    sampleString(seed, n, temperature) {
        seed = seed || this.randomCharacter();

	this.resetStates();
        this.advance(seed.slice(0, -1));

        var res = seed;
        var c = seed.slice(-1);
        for (var i=0; i < n; i++) {
            c = this.sampleCharacter(c, temperature);
            res += c;
        }

        return res;
    }

    sampleCharacter(current_char, temperature) {
        const i = this.encodeChar(current_char);
        const q = this.model.predict(tf.tensor([[i]]));
        const pt = tf.exp(tf.log(tf.maximum(q, 1e-40)).div(tf.scalar(temperature)));
        const ptNorm = pt.norm(1);
        var p;
        if (ptNorm.arraySync() > 1e-16) {
            p = pt.div(ptNorm).squeeze();
        } else {
            // Numerical stability: Assign all probability on the
            // maximum element
            p = tf.oneHot(tf.argMax(q.flatten()), Object.keys(this.char2idx).length)
                .asType('float32');
        }

        return this.idx2char[
            tf.multinomial(p, 1, undefined, true).arraySync()
        ];
    }

    resetStates() {
	this.model.resetStates();
    }

    randomCharacter() {
        const characters = Object.keys(this.char2idx).join('');
        return characters.charAt(Math.floor(Math.random() * characters.length));
    }

    advance(text) {
        for (var c of text) {
            const i = this.encodeChar(c);
            this.model.predict(tf.tensor([[i]]));
        }
    }

    encodeText(text) {
        return text.map(this.encodeChar);
    }

    encodeChar(c) {
        return this.char2idx[c] || this.char2idx[' '];
    }

    reverseChar2idx(char2idx) {
        const idx2char = {};
        for (var c in char2idx) {
            idx2char[char2idx[c]] = c;
        }
        return idx2char;
    }
}

function generateVerses(sampler, keywords) {
    const versesNode = document.getElementById("verses");
    versesNode.innerHTML = '';
    
    const verses = sampler.sampleString(undefined, 120, 0.1).split('\n');
    for (const line of verses) {
        versesNode.appendChild(document.createTextNode(line));
        versesNode.appendChild(document.createElement('br'));
        versesNode.appendChild(document.createTextNode('\n'));
    }
}

async function runKalevalaSampler() {
    const char2idx = await (await fetch('/tfjs/char2idx.json')).json();
    const model = await tf.loadLayersModel('/tfjs/model.json');
    const sampler = new TextSampler(model, char2idx);

    function generateWithKeywords() {
        const verses = document.getElementById('keywords-input').value.split(' ');
        document.getElementById("keywords-input").value = "";

        generateVerses(sampler, verses);
    };

    function keyhandler(event) {
        if (event.key == "Enter" || event.keyCode == 13) {
            generateWithKeywords();
            return false;
        } else {
            return true;
        }
    }

    document.getElementById('btn-generate').addEventListener('click', generateWithKeywords);
    document.getElementById('keywords-input').addEventListener('keydown', keyhandler);

    generateVerses(sampler, verses);
}

runKalevalaSampler();
