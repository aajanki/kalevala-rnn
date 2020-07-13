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
    // Seed is an optional string that will be the start of the sequence.
    async* verseGenerator(temperature, seed) {
        const characters = await this.characterGenerator(temperature, seed);

        for await (const lineCharacters of asyncSplitOn('\n', characters)) {
            const charArray = await asyncToArray(lineCharacters);
            const line = charArray.join('') + '\n';
            yield line;
        }
    }

    async* characterGenerator(temperature, seed) {
        seed = seed || this.randomUpperCaseCharacter();

        // Initialize the internal state
        this.model.resetStates();
        this.advance(seed.slice(0, -1));

        yield* seed;
        
        var c = seed.slice(-1);
        while (true) {
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

async function replaceVerses(sampler, keywords) {
    const versesNode = document.getElementById("verses");
    versesNode.innerHTML = '';

    const verses = await sampler.verseGenerator(0.1);
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

async function initialize() {
    const char2idx = await (await fetch('/tfjs/char2idx.json')).json();
    const model = await tf.loadLayersModel('/tfjs/model.json');
    const sampler = new TextSampler(model, char2idx);

    async function generateWithKeywords() {
        const verses = document.getElementById('keywords-input').value.split(' ');
        document.getElementById("keywords-input").value = "";

        await replaceVerses(sampler, verses);
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
        .addEventListener('click', async () => await generateWithKeywords());
    document
        .getElementById('keywords-input')
        .addEventListener('keydown', async () => await keyhandler());

    await replaceVerses(sampler, verses);
}

initialize();
