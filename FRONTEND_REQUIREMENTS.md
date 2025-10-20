# Multi-Tenant AI Chatbot - Frontend Requirements (Next.js/React)

## Project Overview
Build a modern, responsive Multi-Tenant AI Chatbot frontend using **Next.js 14+ (App Router)** or **React with Vite**. The application is a sophisticated RAG (Retrieval-Augmented Generation) chatbot with organization management, document upload, website ingestion, and real-time streaming chat capabilities.

---


## Backend API Base URL
- Production: `https://zibtek-chatbot-42bb2cdc74d2.herokuapp.com`
---

## Core Features & Components Breakdown

---

## 1. AUTHENTICATION SYSTEM

### Components Needed:

#### **LoginPage** (`/login` or `/auth/login`)
**Features:**
- Email + password form
- Form validation (email format, password minimum 6 chars)
- "Remember me" checkbox
- Error handling for invalid credentials
- Loading state during authentication
- Redirect to dashboard after successful login

**API Endpoint:**
```
POST /auth/login
Body: { email: string, password: string }
Response: { user: {...}, token: string }
```

#### **SignupPage** (`/signup` or `/auth/signup`)
**Features:**
- Email, password, confirm password, full name (optional) fields
- Real-time password validation
- Password strength indicator
- Password match validation
- Error handling for existing email
- Auto-login after successful signup

**API Endpoint:**
```
POST /auth/signup
Body: { email: string, password: string, full_name?: string }
Response: { user: {...}, token: string }
```

#### **AuthContext/Provider**
**State:**
- `user`: { id, email, full_name, created_at }
- `token`: string (JWT token)
- `isAuthenticated`: boolean
- `isLoading`: boolean

**Methods:**
- `login(email, password)`
- `signup(email, password, fullName)`
- `logout()`
- `refreshUser()`

**Features:**
- Store token in localStorage/sessionStorage
- Attach token to all API requests via Authorization header
- Auto-redirect to login on 401 responses
- Persist auth state across page refreshes
- Protected route wrapper component

---

## 2. MAIN CHAT INTERFACE

### Components Needed:

#### **ChatPage** (`/chat` or `/`)
Main chat interface with sidebar and message area.

**Layout Structure:**
```
┌──────────────────────────────────────────────────────┐
│  Sidebar (collapsible)  │    Chat Area              │
│  - User Profile         │    - Chat Header          │
│  - Organization Select  │    - Message History      │
│  - Conversation List    │    - Message Input        │
│  - New Chat Button      │                           │
└──────────────────────────────────────────────────────┘
```

#### **Sidebar Component**
**Features:**
- User profile display (email, avatar)
- Logout button
- Organization/Namespace selector dropdown
- Current namespace badge display
- "New Conversation" button
- Conversation history list (last 10 for current namespace)
- Delete conversation button for each conversation
- Refresh button to reload data
- Responsive: collapsible on mobile

**State:**
- `currentNamespace`: string
- `conversations`: grouped by namespace
- `organizations`: array
- `websites`: array

**API Endpoints:**
```
GET /organizations → Array of organizations
GET /websites → Array of websites
GET /conversations → Array of user's conversations
DELETE /conversations/{namespace}/{sessionId}
```

#### **NamespaceSelector Component**
**Features:**
- Dropdown/Select showing:
  - "Zibtek (default)" → namespace: "zibtek"
  - Each website as: "{org_name} - {website_url}" → namespace from website
- Change namespace triggers new conversation
- Visual indicator of current selection
- Search/filter if many organizations

#### **ConversationList Component**
**Features:**
- List of conversations for current namespace
- Each item shows: title (truncated to 30 chars), timestamp
- Click to load conversation
- Delete button (confirmation modal)
- Empty state message
- Loading skeleton during fetch

#### **ChatMessageList Component**
**Features:**
- Scrollable message container
- Auto-scroll to bottom on new message
- Render user messages (right-aligned, different styling)
- Render assistant messages (left-aligned)
- Empty state: "👋 Hello! Ask me anything..."
- Loading indicator for streaming

#### **MessageBubble Component**
**Props:** `message` object

**User Message Structure:**
```typescript
{
  content: string
  is_user: true
  timestamp: string (ISO)
}
```

**Assistant Message Structure:**
```typescript
{
  content: string
  is_user: false
  citations?: string[] // Array of URLs
  retrieval_steps?: {...} // Detailed retrieval info
  is_out_of_scope?: boolean
  processing_time_ms?: number
  timestamp: string
}
```

**Features:**
- Markdown rendering for message content
- Citation links (styled as pills/chips)
- Expandable "Retrieval Pipeline Details" section
- Metadata footer: processing time, grounded status, timestamp
- Different styling for grounded vs out-of-scope responses
- Streaming typing indicator

#### **CitationsList Component**
**Features:**
- Render citation URLs as clickable chips/buttons
- Extract domain name for display
- Open links in new tab
- Styled with icons (🔗)
- Responsive grid layout


#### **ChatInput Component**
**Features:**
- Text input with placeholder
- "Send" button
- Auto-focus on mount
- Disable during streaming
- Enter key to send (Shift+Enter for new line)
- Character counter (max 1000 chars)
- Clear input after send
- Loading state with spinner

**Chat Streaming Logic:**
- Use **Server-Sent Events (SSE)** or manual stream parsing
- Stream chunks and update message in real-time
- Handle stream events:
  - `type: "token"` → append to current message
  - `type: "complete"` → show citations and metadata
  - `type: "error"` → show error message

**API Endpoint:**
```
POST /chat/stream
Body: {
  question: string
  session_id: string
  partition_name: string (namespace)
}
Headers: { Authorization: "Bearer {token}" }

Stream Response (SSE):
data: {"type": "token", "content": "partial text"}
data: {"type": "complete", "citations": [...], "processing_time_ms": 123, ...}
data: {"type": "error", "content": "error message"}
```

**Streaming Implementation Example:**
```typescript
const response = await fetch('/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
  body: JSON.stringify({ question, session_id, partition_name })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.type === 'token') {
        // Update streaming message
      } else if (data.type === 'complete') {
        // Finalize message with citations
      }
    }
  }
}
```

---

## 3. ORGANIZATION MANAGEMENT

### Components Needed:

#### **OrganizationsPage** (`/organizations`)
**Tab Structure:**
1. Organizations Tab
2. Websites Tab
3. Ingestion Jobs Tab

#### **OrganizationsTab Component**
**Features:**
- "Create New Organization" expandable form
  - Organization name (required)
  - Description (optional, textarea)
  - Submit button
- List of existing organizations
  - Organization card showing: name, slug, ID, description, created date
  - No organizations → empty state message

**API Endpoints:**
```
GET /organizations
POST /organizations
Body: { name: string, description?: string }
```

#### **WebsitesTab Component**
**Features:**
- "Add New Website" expandable form
  - Organization selector (dropdown)
  - Website URL (text input with validation)
  - Submit button
- Filter dropdown: "All" or filter by organization
- List of websites
  - Website card showing:
    - URL
    - Namespace badge
    - Status indicator (pending 🟡, ingesting 🔵, completed 🟢, failed 🔴)
    - Pages crawled count
    - Chunks created count
    - "Start Ingestion" button (conditional based on status)

**API Endpoints:**
```
GET /websites?org_id={optional}
POST /organizations/{org_id}/websites
Body: { org_id: string, url: string }

POST /websites/{website_id}/ingest?max_pages=500
Response: { job_id: string, ... }
```

#### **IngestionJobsTab Component**
**Features:**
- Auto-refresh checkbox (every 5 seconds)
- List of ingestion jobs (latest 20)
- Job card showing:
  - Website URL
  - Job ID (truncated)
  - Status icon and badge
  - Progress bar (if running/pending)
  - Pages crawled metric
  - Chunks created metric
  - Error message (if failed)
- Auto-refresh logic when active jobs exist
- Polling mechanism for job status

**API Endpoints:**
```
GET /websites/{website_id}/jobs → Array of jobs
GET /jobs/{job_id}/status → Runtime status
```

**Job Status Polling:**
- Poll `/jobs/{job_id}/status` every 5 seconds for active jobs
- Update progress bar and metrics in real-time
- Stop polling when all jobs complete

---

## 4. DOCUMENT UPLOAD

### Components Needed:

#### **DocumentsPage** (`/documents`)
**Features:**
- Title and description
- Auth check (redirect if not logged in)
- Organization selector dropdown
- Current namespace info badge
- Website list for selected organization (expandable)
- File upload dropzone
- File information display (name, size, type)
- File size validation (max 10MB)
- Upload button with loading state
- Upload results display:
  - Chunks created
  - Characters processed
  - Time taken
  - Document ID
  - Success message
- "Upload Another" button
- List of uploaded documents for selected org
  - Document cards with: filename, status, chunks, size, upload date, doc ID
  - Expandable to show details
- Information sections (expandable):
  - Supported formats
  - File size limits
  - How it works
  - Privacy & security

**API Endpoints:**
```
GET /documents/org/{org_id} → Array of documents
POST /documents/upload
Body: FormData { file: File, org_id: string }
Response: {
  success: boolean
  doc_id: string
  chunks: number
  characters: number
  elapsed_seconds: number
  namespace: string
}
```

#### **FileUploader Component**
**Features:**
- Drag-and-drop zone
- Click to browse
- File type restriction: .pdf, .txt, .docx
- File size validation
- Preview before upload
- Progress indicator during upload
- Error handling

#### **DocumentCard Component**
**Features:**
- Status badge with emoji
- Metrics grid (chunks, size, date)
- Document ID display
- Error message display (if any)
- Expandable details section

---

## 5. NAVIGATION & LAYOUT

### Components Needed:

#### **AppLayout** (Layout wrapper for authenticated pages)
**Features:**
- Top navigation bar
  - Logo/app name
  - Navigation links (Chat, Documents, Organizations)
  - User menu dropdown (logout)
- Main content area
- Responsive sidebar toggle (mobile)

#### **ProtectedRoute Component**
**Features:**
- Check authentication status
- Redirect to /login if not authenticated
- Show loading spinner during auth check
- Wrapper for all protected pages

#### **Navigation Component**
**Links:**
- `/` or `/chat` - Main Chat
- `/documents` - Document Upload
- `/organizations` - Organization Management
- Active link highlighting
- Mobile hamburger menu

---

## 6. UTILITY COMPONENTS & FEATURES

### Components:

#### **LoadingSpinner**
- Full-page loading overlay
- Inline spinner variants
- Skeleton loaders for lists/cards

#### **ErrorBoundary**
- Catch React errors
- Display fallback UI
- Error logging

#### **Toast/Notification System**
- Success, error, warning, info variants
- Auto-dismiss
- Position: top-right or bottom-right

#### **Modal/Dialog Component**
- Confirmation dialogs (delete conversation)
- Generic modal wrapper
- Close on overlay click
- Escape key to close

#### **EmptyState Component**
- Reusable for empty lists
- Custom icon, title, description
- Call-to-action button

#### **ConfirmDialog Component**
- Title, message, confirm/cancel buttons
- Async confirm action
- Loading state during action

---

## 7. GLOBAL STATE STRUCTURE

### AuthContext State:
```typescript
{
  user: {
    id: string
    email: string
    full_name?: string
    created_at: string
  } | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
}
```

### ChatContext/State:
```typescript
{
  currentNamespace: string
  currentSessionId: string
  chatHistory: Message[]
  conversations: Record<string, Conversation[]>
  organizations: Organization[]
  websites: Website[]
}
```

### Types:
```typescript
interface Message {
  content: string
  is_user: boolean
  timestamp: string
  citations?: string[]
  retrieval_steps?: RetrievalSteps
  is_out_of_scope?: boolean
  processing_time_ms?: number
}

interface Conversation {
  id: string
  session_id: string
  namespace: string
  title: string
  created_at: string
  updated_at: string
}

interface Organization {
  id: string
  name: string
  slug: string
  description?: string
  created_at: string
}

interface Website {
  id: string
  org_id: string
  url: string
  namespace: string
  status: 'pending' | 'ingesting' | 'completed' | 'failed'
  pages_crawled?: number
  chunks_created?: number
  created_at: string
}

interface Document {
  doc_id: string
  filename: string
  file_type: string
  file_size_bytes: number
  chunk_count: number
  namespace: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  uploaded_at: string
  error_message?: string
}

interface IngestionJob {
  id: string
  website_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress_percent: number
  pages_crawled: number
  chunks_created: number
  error_message?: string
  created_at: string
}
```

### Required Features:
1. **Code Splitting:** Lazy load pages and heavy components
2. **Memoization:** Use React.memo, useMemo, useCallback appropriately
3. **Virtual Scrolling:** For long message lists (react-window)
4. **Debouncing:** Search inputs, auto-save
5. **Optimistic Updates:** Update UI before API response for better UX
6. **Error Boundaries:** Graceful error handling
7. **Loading States:** Skeletons, spinners, progress bars
8. **Empty States:** Clear messaging when no data
9. **Accessibility:** ARIA labels, keyboard navigation, focus management
10. **Dark Mode:** (Optional but recommended)

---

## 11. STYLING GUIDELINES

### Design System:
- **Primary Color:** Blue (#1f77b4)
- **Success:** Green (#10b981)
- **Warning:** Yellow (#f59e0b)
- **Error:** Red (#ef4444)
- **Info:** Blue (#3b82f6)

### Component Styling:
- Rounded corners (rounded-lg, rounded-md)
- Subtle shadows for cards
- Border colors from Tailwind palette
- Consistent spacing (4px increments)
- Smooth transitions (transition-colors, transition-all)

### Typography:
- Font: System font stack or Inter
- Headings: Bold, larger sizes
- Body: Regular weight, 16px base
- Captions: Smaller, muted color

---

## 12. ADDITIONAL FEATURES (Nice-to-Have)

1. **Search within conversations**
2. **Export conversation as PDF/Text**
3. **Conversation sharing (read-only link)**
4. **User settings page** (profile, preferences)
5. **Admin dashboard** (analytics, usage stats)
6. **Real-time collaboration** (multiple users in same org)
7. **Voice input** for chat
8. **Code syntax highlighting** in messages
9. **File preview** before upload
10. **Bulk document upload**
11. **Document delete functionality**
12. **Website crawl depth configuration**
13. **Custom prompt templates**
14. **Feedback system** (thumbs up/down on responses)

---

## 13. TESTING REQUIREMENTS

### Unit Tests:
- Utility functions
- Form validation logic
- Context providers
- Custom hooks

### Integration Tests:
- API integration
- Authentication flow
- Chat message flow
- Document upload flow

### E2E Tests (Playwright/Cypress):
- Complete user journeys
- Login → Chat → Document Upload
- Organization creation → Website ingestion

---

## 14. DEPLOYMENT CONSIDERATIONS

### Environment Variables:
```env
NEXT_PUBLIC_API_BASE_URL=https://zibtek-chatbot-42bb2cdc74d2.herokuapp.com
NEXT_PUBLIC_APP_NAME=Zibtek AI Chatbot
NEXT_PUBLIC_ENABLE_ANALYTICS=false
```

### Build Optimization:
- Enable compression
- Image optimization (Next.js Image component)
- Bundle size monitoring
- Performance budgets

### Hosting Options:
- **Vercel** (for Next.js)
- **Netlify** (for React)
- **AWS Amplify**
- **Cloudflare Pages**

---

## 15. FOLDER STRUCTURE RECOMMENDATION

```
src/
├── app/ (Next.js App Router)
│   ├── (auth)/
│   │   ├── login/
│   │   └── signup/
│   ├── (main)/
│   │   ├── layout.tsx
│   │   ├── page.tsx (chat)
│   │   ├── documents/
│   │   └── organizations/
│   └── layout.tsx
├── components/
│   ├── chat/
│   │   ├── ChatMessageList.tsx
│   │   ├── MessageBubble.tsx
│   │   ├── ChatInput.tsx
│   │   ├── CitationsList.tsx
│   │   └── RetrievalSteps.tsx
│   ├── organizations/
│   │   ├── OrganizationCard.tsx
│   │   ├── WebsiteCard.tsx
│   │   └── IngestionJobCard.tsx
│   ├── documents/
│   │   ├── FileUploader.tsx
│   │   └── DocumentCard.tsx
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   ├── Navbar.tsx
│   │   └── AppLayout.tsx
│   └── ui/ (shadcn components)
│       ├── button.tsx
│       ├── input.tsx
│       ├── card.tsx
│       ├── dialog.tsx
│       └── ...
├── contexts/
│   ├── AuthContext.tsx
│   └── ChatContext.tsx
├── hooks/
│   ├── useAuth.ts
│   ├── useChat.ts
│   ├── useOrganizations.ts
│   └── useDocuments.ts
├── lib/
│   ├── api-client.ts
│   ├── queries.ts (React Query)
│   └── utils.ts
├── types/
│   └── index.ts
└── styles/
    └── globals.css
```

---

## 16. FINAL PROMPT SUMMARY

**Build a modern Multi-Tenant AI Chatbot frontend with:**

✅ **Authentication:** Login/Signup with JWT token management
✅ **Real-time Chat:** Streaming responses with SSE, markdown rendering, citations
✅ **Organization Management:** Create orgs, add websites, trigger ingestion
✅ **Document Upload:** Upload PDFs/TXT/DOCX with progress tracking
✅ **Conversation History:** Load/delete past conversations
✅ **Multi-tenant Support:** Namespace-based data isolation
✅ **Retrieval Pipeline Visibility:** Show vector search, BM25, reranking steps
✅ **Responsive Design:** Mobile-first with Tailwind CSS
✅ **Type Safety:** TypeScript throughout
✅ **State Management:** React Query for server state, Context for auth/global state
✅ **Component Library:** shadcn/ui for consistent, accessible UI
✅ **Error Handling:** Graceful errors, loading states, empty states
✅ **Performance:** Code splitting, memoization, optimistic updates

**Key Technical Decisions:**
- Framework: Next.js 14 (App Router) preferred
- Styling: Tailwind CSS + shadcn/ui
- Data Fetching: TanStack Query (React Query)
- Forms: React Hook Form + Zod
- Icons: Lucide React
- Notifications: sonner or react-hot-toast
- Streaming: Manual SSE parsing for chat

**Priority Features:**
1. Authentication system (HIGH)
2. Main chat interface with streaming (HIGH)
3. Sidebar with conversations (HIGH)
4. Organization management (MEDIUM)
5. Document upload (MEDIUM)
6. Ingestion jobs monitoring (LOW)
7. Retrieval pipeline details (LOW)

Build incrementally, starting with auth and chat, then expanding to other features.

---

## API ENDPOINTS QUICK REFERENCE

### Auth
- `POST /auth/login` - Login
- `POST /auth/signup` - Signup

### Chat
- `POST /chat/stream` - Streaming chat (SSE)

### Conversations
- `GET /conversations` - List user conversations
- `GET /conversations/{namespace}/{sessionId}/messages` - Get messages
- `DELETE /conversations/{namespace}/{sessionId}` - Delete conversation

### Organizations
- `GET /organizations` - List all orgs
- `POST /organizations` - Create org

### Websites
- `GET /websites?org_id={optional}` - List websites
- `POST /organizations/{orgId}/websites` - Add website
- `POST /websites/{websiteId}/ingest?max_pages=500` - Start ingestion

### Documents
- `GET /documents/org/{orgId}` - List documents
- `POST /documents/upload` - Upload document (multipart/form-data)

### Jobs
- `GET /websites/{websiteId}/jobs` - List jobs
- `GET /jobs/{jobId}/status` - Get runtime status

---

**Good luck building! 🚀**
