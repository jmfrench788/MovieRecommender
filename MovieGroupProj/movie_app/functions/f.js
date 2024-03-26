function getSaved(){
        moviesFile = localStorage.getItem('selMovies') || '';
        const obj = JSON.parse(moviesFile);
        console.log(obj)
}
