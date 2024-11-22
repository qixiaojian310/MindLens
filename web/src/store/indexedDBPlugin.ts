import type { PiniaPluginContext } from 'pinia'

const DATABASE_NAME = 'PiniaDB'
const STORE_NAME = 'PiniaStore'
const DATABASE_VERSION = 1

// 定义返回 IndexedDB 数据库的类型
function openIndexedDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION)

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME)
      }
    }

    request.onsuccess = (event) => {
      resolve((event.target as IDBOpenDBRequest).result)
    }

    request.onerror = (event) => {
      reject((event.target as IDBOpenDBRequest).error)
    }
  })
}

// 定义保存到 IndexedDB 的函数
function saveToIndexedDB<T>(key: string, value: T): Promise<void> {
  return openIndexedDB().then((db) => {
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readwrite')
      const store = transaction.objectStore(STORE_NAME)
      const request = store.put(value, key)

      request.onsuccess = () => resolve()
      request.onerror = event => reject((event.target as IDBRequest).error)
    })
  })
}

// 定义从 IndexedDB 加载数据的函数
function loadFromIndexedDB<T>(key: string): Promise<T | undefined> {
  return openIndexedDB().then((db) => {
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readonly')
      const store = transaction.objectStore(STORE_NAME)
      const request = store.get(key)

      request.onsuccess = () => resolve(request.result as T)
      request.onerror = event => reject((event.target as IDBRequest).error)
    })
  })
}

// Pinia 插件
export function indexedDBPlugin(context: PiniaPluginContext): void {
  const { store } = context
  const key = store.$id // 使用 store 的 ID 作为主键

  // 初始化时从 IndexedDB 加载数据
  loadFromIndexedDB<Partial<typeof store.$state>>(key).then((data) => {
    if (data) {
      store.$patch(data) // 恢复状态
    }
  })

  // 监听状态变化并持久化到 IndexedDB
  store.$subscribe((_, state) => {
    const serializableState = JSON.parse(JSON.stringify(state)) // 确保去除 Proxy
    saveToIndexedDB(key, serializableState).catch((err) => {
      console.error(`保存状态到 IndexedDB 时出错: ${err}`)
    })
  })
}
