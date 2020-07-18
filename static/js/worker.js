import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import {setWasmPath} from '@tensorflow/tfjs-backend-wasm';
import {
    asyncDrop, asyncDropWhile, asyncSplitOn, asyncToArray, pipe
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

    async sampleCharacter(currentChar, temperature) {
        const prob = tf.tidy(() => {
            const i = this.encodeChar(currentChar);
            const q = this.model.predict(tf.tensor([[i]])).squeeze();

            // Scale with temperature in the log space and normalize so
            // that probabilities sum to one in the linear space.
            const logQt = q.maximum(1e-37).log().div(temperature);
            const scaling = tf.logSumExp(logQt);
            return logQt.sub(scaling).exp();
        })

        // The wasm backend doesn't support tf.multinomial. Therefore,
        // we use a plain Javascript implementation.
        const nextChar = sampleWeighted(Object.values(this.idx2char), await prob.array());
        prob.dispose();

        return nextChar;
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
        const i = this.char2idx[c];
        if (typeof i === 'undefined') {
            return this.char2idx[' '];
        } else {
            return i;
        }
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

/*
 * asyncTake from iter-tools actually consumes n+1 elements from the
 * iterable. This optimized version consumes exactly n elements.
 */
async function* asyncTake(n, iterable) {
    if (n > 0) {
        let i = 0;
        for await (const item of iterable) {
            yield item;
            i++;
            if (i >= n) break;
        }
    }
}

async function loadModel() {
    await tf.ready();
    console.log(`Tensorflow backend: ${tf.getBackend()}`);

    const char2idx = await (await fetch('/tfjs/char2idx.json')).json();
    const model = await tf.loadLayersModel('/tfjs/model.json');
    const sampler = new TextSampler(model, char2idx);

    return sampler;
}

async function sampleVerses(sampler, keywords) {
    const t0 = performance.now();

    const seeds = new SeedKeywords(undefined, keywords);
    const verseIter = await sampler.verseGenerator(0.1, seeds);
    const selectVerses = it => asyncTake(
        6,
        pipe(
            asyncDrop(1),
            asyncDropWhile(s => s == '\n'),
        )(it));

    const versesArray = await asyncToArray(selectVerses(verseIter));

    const t1 = performance.now();
    console.debug(`Generation took ${Math.round(t1 - t0)} ms`);

    return versesArray;
}

let sampler;

onmessage = async function (e) {
    if (typeof sampler === 'undefined') {
        sampler = await loadModel()
    }
    const verses = await sampleVerses(sampler, e.data);

    postMessage(verses);
};

setWasmPath('/tfjs-backend-wasm.wasm');
tf.setBackend('wasm').catch(console.warn);
