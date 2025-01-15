-- Forum Database Schema
-- Note: Some vulnerabilities are intentionally included in this schema

-- Users Table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,git 
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,  -- Intentionally not using proper password hashing
    email VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',  -- Vulnerable to role manipulation
    reputation INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    -- Vulnerable: No input validation on role
    CHECK (role IN ('user', 'moderator', 'admin'))
);

-- Sessions Table (vulnerable to session fixation)
CREATE TABLE sessions (
    id VARCHAR(255) PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    ip_address VARCHAR(45),
    user_agent TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Categories Table
CREATE TABLE categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parent_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER,
    is_hidden BOOLEAN DEFAULT false,  -- Hidden categories feature
    FOREIGN KEY (parent_id) REFERENCES categories(id),
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- Threads Table
CREATE TABLE threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,  -- Vulnerable to XSS
    author_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    is_locked BOOLEAN DEFAULT false,
    is_sticky BOOLEAN DEFAULT false,
    view_count INTEGER DEFAULT 0,
    visibility VARCHAR(20) DEFAULT 'public',  -- Vulnerable to visibility manipulation
    deleted_at TIMESTAMP,  -- Soft delete with potential data leak
    FOREIGN KEY (category_id) REFERENCES categories(id),
    FOREIGN KEY (author_id) REFERENCES users(id)
);

-- Posts Table
CREATE TABLE posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL,
    content TEXT NOT NULL,  -- Vulnerable to XSS
    author_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    edit_count INTEGER DEFAULT 0,
    is_hidden BOOLEAN DEFAULT false,
    parent_post_id INTEGER,  -- For nested replies
    deleted_at TIMESTAMP,  -- Soft delete with potential data leak
    -- Vulnerable: Hidden content in deleted posts
    hidden_content TEXT,  -- Intentionally exposed field
    FOREIGN KEY (thread_id) REFERENCES threads(id),
    FOREIGN KEY (author_id) REFERENCES users(id),
    FOREIGN KEY (parent_post_id) REFERENCES posts(id)
);

-- Votes Table (vulnerable to vote manipulation)
CREATE TABLE votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    vote_type INTEGER NOT NULL,  -- 1 for upvote, -1 for downvote
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Vulnerable: No protection against rapid voting
    FOREIGN KEY (post_id) REFERENCES posts(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- User Reputation History (vulnerable to point manipulation)
CREATE TABLE reputation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    points INTEGER NOT NULL,
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Vulnerable: No validation on points
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Private Messages (vulnerable to message manipulation)
CREATE TABLE private_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_id INTEGER NOT NULL,
    recipient_id INTEGER NOT NULL,
    subject VARCHAR(255),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    read_at TIMESTAMP,
    is_hidden BOOLEAN DEFAULT false,
    -- Hidden moderator messages feature
    is_system_message BOOLEAN DEFAULT false,
    FOREIGN KEY (sender_id) REFERENCES users(id),
    FOREIGN KEY (recipient_id) REFERENCES users(id)
);

-- User Activity Log (vulnerable to timing attacks)
CREATE TABLE user_activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    target_type VARCHAR(50),
    target_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    metadata TEXT,  -- Vulnerable to metadata exploitation
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Search Cache (vulnerable to cache poisoning)
CREATE TABLE search_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    results TEXT NOT NULL,  -- Stored as JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    -- Vulnerable: No validation on cached content
    CHECK (json_valid(results))
);

-- Create indexes
CREATE INDEX idx_threads_category ON threads(category_id);
CREATE INDEX idx_posts_thread ON posts(thread_id);
CREATE INDEX idx_votes_post ON votes(post_id);
CREATE INDEX idx_votes_user ON votes(user_id);
CREATE INDEX idx_activity_user ON user_activity_log(user_id);
CREATE INDEX idx_activity_target ON user_activity_log(target_type, target_id);

-- Create views for common queries (potentially vulnerable to information disclosure)
CREATE VIEW thread_statistics AS
SELECT 
    t.id,
    t.title,
    COUNT(DISTINCT p.id) as post_count,
    COUNT(DISTINCT v.id) as vote_count,
    t.view_count,
    t.created_at,
    t.updated_at
FROM threads t
LEFT JOIN posts p ON t.id = p.thread_id
LEFT JOIN votes v ON p.id = v.post_id
GROUP BY t.id;

-- Create triggers for reputation updates (vulnerable to race conditions)
CREATE TRIGGER update_user_reputation
AFTER INSERT ON reputation_history
BEGIN
    UPDATE users 
    SET reputation = reputation + NEW.points 
    WHERE id = NEW.user_id;
END;