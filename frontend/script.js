async function searchPlayer() {
    const name = document.getElementById("playerSearch").value;
    const response = await fetch(`/search_player?name=${name}`);
    const players = await response.json();

    console.log(players);  // Aqui você preenche sua lista de jogadores

    // Quando clicar em algum:
    fetch("/predict_transfer", {
        method: "POST",
        body: JSON.stringify({ player_id: players[0].player_id }),
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.json())
    .then(result => {
        console.log(result);
        alert(`Chance de ser transferido: ${result.transfer_probability}%`);
    });
}
