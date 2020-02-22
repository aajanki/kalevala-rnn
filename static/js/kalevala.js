function generate_verses(keywords) {
    let url = "/api/verses";
    if (keywords) {
        url = url + "?keywords=" + encodeURIComponent(keywords);
    }

    fetch(url).then(function(response) {
        return response.json();
    }).then(function(verses) {
        let verses_html = verses.join("\n<br>\n");
        document.getElementById("verses").innerHTML = verses_html;

        document.getElementById("keywords-input").value = "";
    });
}

function keyhandler(event) {
    if (event.key == "Enter" || event.keyCode == 13) {
        generate_verses(document.getElementById("keywords-input").value);
        return false;
    } else {
        return true;
    }
}
