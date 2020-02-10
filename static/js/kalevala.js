function generate_verses() {
    fetch("/api/verses").then(function(response) {
	return response.json();
    }).then(function(verses) {
	let verses_html = verses.join("\n<br>\n");
	document.getElementById("poem-text").innerHTML = verses_html;
    });
}
