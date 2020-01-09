
// Get the hash of the url
const hash = window.location.hash
.substring(1)
.split('&')
.reduce(function (initial, item) {
  if (item) {
    var parts = item.split('=');
    initial[parts[0]] = decodeURIComponent(parts[1]);
  }
  return initial;
}, {});
window.location.hash = '';

// Set token
let _token = hash.access_token;

const authEndpoint = 'https://accounts.spotify.com/authorize';

// Replace with your app's client ID, redirect URI and desired scopes

const clientId = '6be02df553ae4215a978d21e2fe46d4d';
const redirectUri = 'http://127.0.0.1:5500/';
const scopes = [
  'streaming',
  'user-read-private',
  'user-modify-playback-state'
];

// If there is no token, redirect to Spotify authorization
if (!_token) {
  window.location = `${authEndpoint}?client_id=${clientId}&redirect_uri=${redirectUri}&scope=${scopes.join('%20')}&response_type=token&show_dialog=true`;
}

// Set up the Web Playback SDK

window.onSpotifyPlayerAPIReady = () => {
  const player = new Spotify.Player({
    name: 'Adaptive Control Music System Web Playback',
    getOAuthToken: cb => { cb(_token); }
  });

  // Error handling
  player.on('initialization_error', e => console.error(e));
  player.on('authentication_error', e => console.error(e));
  player.on('account_error', e => console.error(e));
  player.on('playback_error', e => console.error(e));

  // Playback status updates
  player.on('player_state_changed', ({
    position,
    duration,
    track_window
  }) => {
    console.log("1");
    $('#current-track').attr('src', track_window.current_track.album.images[0].url);
    $('#current-track-name').text(track_window.current_track.name);
    $('#current-track-duration').text(duration);
    getCurrentState()
  });


  function getCurrentState(){
      player.getCurrentState().then(state => {
      if (!state) {
        console.error('User is not playing music through the Web Playback SDK');
        return;
      }
    
      let {
        current_track,
        next_tracks: [next_track]
      } = state.track_window;
      let {
        paused,
        position,
      } = state;
      //console.log(position);
    
      $('#current-track-position').text(position);
      if (!paused){
          timeoutId = setTimeout(function () {
            getCurrentState();
        }, 100);
      }
    });
  }


  // Ready
  player.on('ready', data => {
    console.log('Ready with Device ID', data.device_id);
    
    // Play a track using our new device ID
    play(data.device_id);
  });

  // Connect to the player!
  player.connect();
}

// Play a specified track on the Web Playback SDK's device ID
function play(device_id) {
  demoSongUri = loadFile("demo_song.txt");
  console.log('Song URI', demoSongUri)
  $.ajax({
   url: "https://api.spotify.com/v1/me/player/play?device_id=" + device_id,
   type: "PUT",
   data: '{"uris": ["'+demoSongUri+'"]}',
   beforeSend: function(xhr){xhr.setRequestHeader('Authorization', 'Bearer ' + _token );},
   success: function(data) { 
     console.log(data)
   }
  });
}

// Synchronously read a text file from the web server with Ajax
//
// The filePath is relative to the web page folder.
// Example:   myStuff = loadFile("Chuuk_data.txt");
//
// You can also pass a full URL, like http://sealevel.info/Chuuk1_data.json, but there
// might be Access-Control-Allow-Origin issues. I found it works okay in Firefox, Edge,
// or Opera, and works in IE 11 if the server is configured properly, but in Chrome it only
// works if the domains exactly match (and note that "xyz.com" & "www.xyz.com" don't match).
// Otherwise Chrome reports an error:
//
//   No 'Access-Control-Allow-Origin' header is present on the requested resource. Origin 'http://sealevel.info' is therefore not allowed access.
//
// That happens even when "Access-Control-Allow-Origin *" is configured in .htaccess,
// and even though I verified the headers returned (you can use a header-checker site like
// http://www.webconfs.com/http-header-check.php to check it). I think it's a Chrome bug.
function loadFile(filePath) {
  var result = null;
  var xmlhttp = new XMLHttpRequest();
  xmlhttp.open("GET", filePath, false);
  xmlhttp.send();
  if (xmlhttp.status==200) {
    result = xmlhttp.responseText;
  }
  return result;
}
