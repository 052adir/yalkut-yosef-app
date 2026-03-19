const CACHE_NAME = 'yalkut-yosef-v3';

// Only precache same-origin resources guaranteed to exist
const PRECACHE_URLS = [
  '/',
];

// Install: pre-cache core shell only
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(PRECACHE_URLS);
    }).then(() => self.skipWaiting())
  );
});

// Activate: clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch: network-first for API calls, stale-while-revalidate for static assets
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Skip non-GET requests
  if (event.request.method !== 'GET') return;

  // Never cache API calls (ask, daily, status, feedback)
  if (url.pathname.startsWith('/ask') ||
      url.pathname.startsWith('/daily') ||
      url.pathname.startsWith('/status') ||
      url.pathname.startsWith('/feedback')) {
    return;
  }

  // Cache-first for static assets, with network fallback
  event.respondWith(
    caches.match(event.request).then((cached) => {
      const fetchPromise = fetch(event.request).then((response) => {
        // Only cache successful responses
        if (response && response.status === 200) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, clone);
          });
        }
        return response;
      }).catch(() => null);

      // Return cached version immediately, update in background
      return cached || fetchPromise;
    }).catch(() => {
      // Offline fallback for navigation
      if (event.request.mode === 'navigate') {
        return caches.match('/');
      }
    })
  );
});
