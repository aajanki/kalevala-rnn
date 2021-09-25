const worker = new Worker('./worker.js');

worker.onmessage = function(e) {
    const versesNode = document.getElementById("verses");
    const verses = e.data;

    hideSpinner();

    for (let verse of verses) {
	versesNode.appendChild(document.createTextNode(verse));
	versesNode.appendChild(document.createElement('br'));
	versesNode.appendChild(document.createTextNode('\n'));
    }
};

worker.onerror = function(e) {
    const versesNode = document.getElementById("verses");

    hideSpinner();

    versesNode.innerHTML = '';
    versesNode.appendChild(document.createTextNode('Jotain meni pieleen!'));
    versesNode.appendChild(document.createElement('br'));
    versesNode.appendChild(document.createTextNode('Nyt ei runosuoni syki!'));
};

function replaceVerses(keywords) {
    const versesNode = document.getElementById("verses");
    versesNode.innerHTML = '';
    showSpinner();

    worker.postMessage(keywords || []);
}

function showSpinner() {
    for (const el of document.getElementsByClassName('spinner')) {
        el.style.display = 'block';
    }
}

function hideSpinner() {
    for (const el of document.getElementsByClassName('spinner')) {
        el.style.display = 'none';
    }
}

function generateWithKeywords() {
    const keywords = document.getElementById('keywords-input')
          .value
          .split(/[^a-zåäöA-ZÅÄÖ]+/)
          .filter(s => s.length > 1 && s.length < 15)
          // The RNN seems to output slightly better verses if the line
          // starts with a capital letter like in the source material.
          .map(s => s[0].toUpperCase() + s.slice(1));
    document.getElementById("keywords-input").value = "";

    replaceVerses(keywords);
};

function keyhandler(event) {
    if (event.key == "Enter" || event.keyCode == 13) {
        generateWithKeywords();
        return false;
    } else {
        return true;
    }
}

function initialize() {
    document
        .getElementById('btn-generate')
        .addEventListener('click', generateWithKeywords);
    document
        .getElementById('keywords-input')
        .addEventListener('keydown', keyhandler);

    replaceVerses();
}

initialize();
