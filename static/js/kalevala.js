function generate_verses(keywords) {
    let url = "/api/verses";
    if (keywords) {
        url = url + "?keywords=" + encodeURIComponent(keywords);
    }

    document.getElementById("keywords-input").value = "";

    return fetch(url).then(function(response) {
        if (!response.ok) {
            throw new Error("API returned an error");
        }

        return response.json();
    }).then(function(verses) {
        document.getElementById("verses").innerHTML = verses.join("\n<br>\n");
    }).catch(function(error) {
        document.getElementById("verses").innerHTML = "Hups! Jotain meni pieleen!";
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
