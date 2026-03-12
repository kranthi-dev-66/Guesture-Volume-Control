function startCamera() {

fetch("/start")
.then(response => response.json())
.then(data => {

console.log(data)

document.getElementById("video").src = "/video"

})

}

function stopCamera(){

fetch("/stop")

document.getElementById("video").src = ""

}

setInterval(()=>{

fetch("/volume")
.then(res=>res.json())
.then(data=>{

document.getElementById("vol").innerText=data.volume

let v = Math.min(100,data.volume)

document.getElementById("vol").innerText = v
document.getElementById("progress").style.width = v + "%"
})

},500)