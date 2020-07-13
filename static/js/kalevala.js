import * as tf from '@tensorflow/tfjs';
import {
    asyncDrop, asyncDropWhile, asyncTake, asyncTakeWhile, asyncForEach,
    asyncJoinAsStringWith, asyncTap, asyncSplitOn, asyncToArray, pipe
} from 'iter-tools/es2018';


class TextSampler {
    constructor(model, char2idx) {
        this.model = model;
        this.char2idx = char2idx;
        this.idx2char = this.reverseChar2idx(char2idx);
    }

    // A generator that outputs an infinite sequence of Kalevala verses.
    //
    // Temperature controls the randomness of the output.
    // Seed is a SeedKeywords instance containing potential line prefixes.
    async* verseGenerator(temperature, seeds) {
        const characters = await this.characterGenerator(temperature, seeds);

        for await (const lineCharacters of asyncSplitOn('\n', characters)) {
            const charArray = await asyncToArray(lineCharacters);
            const line = charArray.join('') + '\n';
            yield line;
        }
    }

    async* characterGenerator(temperature, seeds) {
        const prefix = seeds.prefix || this.randomUpperCaseCharacter();

        // Initialize the internal state
        this.model.resetStates();
        this.advance(prefix.slice(0, -1));

        yield* prefix;

        let c = prefix.slice(-1);
        while (true) {
            if (c === '\n') {
                const kw = seeds.sampleLineStartKeyword();
                if (kw) {
                    this.advance('\n');
                    this.advance(kw.slice(0, -1));
                    c = kw.slice(-1);
                    yield* kw;
                }
            }
            
            c = await this.sampleCharacter(c, temperature);
            yield c;
        }
    }

    async sampleCharacter(current_char, temperature) {
        const i = this.encodeChar(current_char);
        const q = this.model.predict(tf.tensor([[i]])).squeeze();

        // Scale with temperature in the log space and normalize so
        // that probabilities sum to one in the linear space.
        const logQt = q.maximum(1e-37).log().div(temperature);
        const scaling = tf.logSumExp(logQt);
        const logP = logQt.sub(scaling);

        return this.idx2char[
            await tf.multinomial(logP, 1).array()
        ];
    }

    randomUpperCaseCharacter() {
        const upperCaseLetters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ';
        const characters = Object.keys(this.char2idx)
              .filter(c => upperCaseLetters.includes(c))
              .join('');

        return characters.charAt(Math.floor(Math.random() * characters.length));
    }

    advance(text) {
        for (let c of text) {
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
        for (let c in char2idx) {
            idx2char[char2idx[c]] = c;
        }
        return idx2char;
    }
}

class SeedKeywords {
    constructor(prefix, keywords) {
        this.prefix = prefix;
        this.keywords = keywords;
        this.kw_weights = keywords.map(x => 0.5);
    }

    sampleLineStartKeyword() {
        if ((typeof this.keywords === 'undefined') ||
            (this.kw_weights.length === 0) ||
            (Math.random() > Math.max(...this.kw_weights)))
        {
            return undefined;
        }

        let i = sampleWeighted(range(this.kw_weights.length), this.kw_weights);
        this.kw_weights[i] = Math.max(0.2*this.kw_weights[i], 0.01);
        return this.keywords[i];
    }
}

async function replaceVerses(sampler, keywords) {
    const versesNode = document.getElementById("verses");
    versesNode.innerHTML = '';

    const seeds = new SeedKeywords(undefined, keywords);
    const verses = await sampler.verseGenerator(0.1, seeds);
    const selectVerses = pipe(
        //asyncTap(item => console.log(item)),
        asyncDrop(1),
        asyncDropWhile(s => s == '\n'),
        asyncTake(6),
    );

    asyncForEach(verse => {
        versesNode.appendChild(document.createTextNode(verse));
        versesNode.appendChild(document.createElement('br'));
        versesNode.appendChild(document.createTextNode('\n'));
    }, selectVerses(verses));
}

function sampleWeighted(items, weights) {
    const sum = weights.reduce((x, y) => x + y, 0);
    let r = Math.random() * sum;
    for (let i=0; i<items.length; i++) {
        if (r < weights[i]) {
            return items[i];
        } else {
            r -= weights[i];
        }
    }

    // not reached
    return items[items.length - 1];
}

function range(n) {
    return [...Array(n).keys()];
}

async function initialize() {
    const char2idx = await (await fetch('/tfjs/char2idx.json')).json();
    const model = await tf.loadLayersModel('/tfjs/model.json');
    const sampler = new TextSampler(model, char2idx);

    async function generateWithKeywords() {
        const keywords = document.getElementById('keywords-input')
              .value
              .split(/[^a-zåäöA-ZÅÄÖ]+/)
              .filter(s => s.length > 1 && s.length < 15)
              // The RNN seems to output slightly better verses if the line
              // starts with a capital letter like in the source material.
              .map(s => s[0].toUpperCase() + s.slice(1));
        document.getElementById("keywords-input").value = "";

        await replaceVerses(sampler, keywords);
    };

    async function keyhandler(event) {
        if (event.key == "Enter" || event.keyCode == 13) {
            await generateWithKeywords();
            return false;
        } else {
            return true;
        }
    }

    document
        .getElementById('btn-generate')
        .addEventListener('click', generateWithKeywords);
    document
        .getElementById('keywords-input')
        .addEventListener('keydown', keyhandler);

    await replaceVerses(sampler, []);
}

initialize();
