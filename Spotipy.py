import sys
import spotipy
import spotipy.util as util

''' shows the albums and tracks for a given artist.
'''

def get_artist(name):
    results = sp.search(q='artist:' + name, type='artist')
    items = results['artists']['items']
    if len(items) > 0:
        return items[0]
    else:
        return None

def show_artist_albums(artist):
    albums = []
    results = sp.artist_albums(artist['id'], album_type='album')
    albums.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        albums.extend(results['items'])
    seen = set() # to avoid dups
    albums.sort(key=lambda album:album['name'].lower())
    for album in albums:
        name = album['name']
        if name not in seen:
            print((' ' + name))
            seen.add(name)

if __name__ == '__main__':
    username = '123'
    scope = 'user-library-read'
    token = util.prompt_for_user_token(username, scope)
    sp = spotipy.Spotify(auth=token)

    name = ' '.join(sys.argv[1:])
    artist = get_artist('GARNiDELiA')
    if artist:
        show_artist_albums(artist)
    else:
        print("Can't find that artist")
