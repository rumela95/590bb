import hashlib
import re
import requests


_VALID_URL = r'https?://(?P<host>(?:news|headlines)\.yahoo\.co\.jp)[^\d]*(?P<id>\d[\d-]*\d)?'

# More functions here...

def yahoojnews_extract(url):
    mobj = re.match(_VALID_URL, url)
    if not mobj:
        raise ValueError('Invalid url %s' % url)
    host = mobj.group('host')
    display_id = mobj.group('id') or host
    webpage = _download_webpage(url)

    title = _search_title(webpage)
    
    if display_id == host:
        # Headline page (w/ multiple BC playlists) ('news.yahoo.co.jp', 'headlines.yahoo.co.jp/videonews/', ...)
        return _playlist_result(webpage)

    # Article page
    description = _search_description(webpage)
    thumbnail = _search_thumbnail(webpage)

    space_id = _search_regex([
            r'<script[^>]+class=["\']yvpub-player["\'][^>]+spaceid=([^&"\']+)',
            r'YAHOO\.JP\.srch\.\w+link\.onLoad[^;]+spaceID["\' ]*:["\' ]+([^"\']+)',
            r'<!--\s+SpaceID=(\d+)'
        ], webpage, 'spaceid')
    content_id = re.search(
        r'<script[^>]+class=(["\'])yvpub-player\1[^>]+contentid=(?P<contentid>[^&"\']+)',
        webpage,
    ).group('contentid')

    r = requests.get(
        'https://feapi-yvpub.yahooapis.jp/v1/content/%s' % content_id,
        headers={
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Origin': 'https://s.yimg.jp',
            'Host': 'feapi-yvpub.yahooapis.jp',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
            'Referer': 'https://s.yimg.jp/images/yvpub/player/vamos/pc/latest/player.html',
        },
        params={
            'appid': 'gj0zaiZpPVZMTVFJR0F...VycbVjcmV0jng9Yju-',
            'output': 'json',
            'space_id': space_id,
            'domain': host,
            'ak': hashlib.md5('_'.join((space_id, host)).encode()).hexdigest(),
            'device_type': '1100',
        },
    )
    r.raise_for_status()
    json_data = r.json()

    formats = _parse_formats(json_data)

    return {
        'id': display_id,
        'title': title,
        'description': description,
        'thumbnail': thumbnail,
        'formats': formats,
    }

yahoojnews_extract("https://news.yahoo.com/michelle-obama-melania-trump-peaceful-transition-not-a-game-white-house-151409430.html")