// Configuração da API do RapidAPI
const RAPID_API_KEY = "998ed467c9msh8cadb7dd850e39fp1dfdfjsne317cbe80c2a";
const RAPID_API_URL = "https://YOUR_ENDPOINT_HERE/football-current-live";

// Função para buscar jogos ao vivo
async function fetchLiveGames() {
    try {
        const response = await fetch(RAPID_API_URL, {
            method: "GET",
            headers: {
                "X-RapidAPI-Key": RAPID_API_KEY,
                "X-RapidAPI-Host": "rapidapi.com"
            }
        });

        const data = await response.json();

        if (data && data.length > 0) {
            document.getElementById("liveGames").innerHTML = data.map(match =>
                `<span>${match.home_team} vs ${match.away_team} - ${match.status}</span>`
            ).join(" | ");
        } else {
            document.getElementById("liveGames").textContent = "No live matches currently.";
        }
    } catch (error) {
        console.error("Error fetching live games:", error);
        document.getElementById("liveGames").textContent = "Failed to load live matches.";
    }
}

// Chamar a função ao carregar a página
fetchLiveGames();
