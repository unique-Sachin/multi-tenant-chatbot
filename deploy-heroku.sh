#!/bin/bash
# Quick deployment script for Heroku

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Heroku Deployment Script${NC}"
echo "================================"

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo -e "${RED}❌ Heroku CLI not found. Please install it first:${NC}"
    echo "brew tap heroku/brew && brew install heroku"
    exit 1
fi

# Check if logged in
if ! heroku auth:whoami &> /dev/null; then
    echo -e "${BLUE}🔐 Logging into Heroku...${NC}"
    heroku login
fi

# Get app name
read -p "Enter your Heroku app name (or press Enter to create new): " APP_NAME

if [ -z "$APP_NAME" ]; then
    echo -e "${BLUE}📦 Creating new Heroku app...${NC}"
    heroku create
    APP_NAME=$(heroku info -s | grep git_url | cut -d'/' -f4 | cut -d'.' -f1)
    echo -e "${GREEN}✅ Created app: ${APP_NAME}${NC}"
else
    # Check if app exists
    if ! heroku info -a "$APP_NAME" &> /dev/null; then
        echo -e "${BLUE}📦 Creating app: ${APP_NAME}${NC}"
        heroku create "$APP_NAME"
    else
        echo -e "${GREEN}✅ Using existing app: ${APP_NAME}${NC}"
    fi
fi

# Add Redis addon
echo -e "${BLUE}🔴 Setting up Redis...${NC}"
if heroku addons:info heroku-redis -a "$APP_NAME" &> /dev/null; then
    echo -e "${GREEN}✅ Redis already configured${NC}"
else
    echo "Adding Redis addon..."
    heroku addons:create heroku-redis:mini -a "$APP_NAME" || \
    heroku addons:create heroku-redis:hobby-dev -a "$APP_NAME"
fi

# Set environment variables
echo -e "${BLUE}🔧 Configuring environment variables...${NC}"

# Check for required variables
echo "Checking for required environment variables..."
REQUIRED_VARS=("OPENAI_API_KEY" "PINECONE_API_KEY" "SUPABASE_URL" "SUPABASE_ANON_KEY" "COHERE_API_KEY" "JWT_SECRET")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if ! grep -q "^${var}=" .env 2>/dev/null; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo -e "${RED}⚠️  Warning: Missing environment variables: ${MISSING_VARS[*]}${NC}"
    echo "Please add them to your .env file before deploying."
fi

# Load from .env file
if [ -f .env ]; then
    echo "Reading from .env file..."
    
    # Extract and set variables (excluding comments and empty lines)
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        if [[ "$line" =~ ^#.*$ ]] || [[ -z "$line" ]]; then
            continue
        fi
        
        # Skip Streamlit auth secret and local Redis URL
        if [[ "$line" =~ ^STREAMLIT_AUTH_SHARED_SECRET.*$ ]] || [[ "$line" =~ ^REDIS_URL=redis://localhost.*$ ]]; then
            continue
        fi
        
        # Extract key and value
        if [[ "$line" =~ ^([^=]+)=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            
            # Map SUPABASE_ANON_KEY to SUPABASE_KEY for consistency
            if [[ "$key" == "SUPABASE_ANON_KEY" ]]; then
                echo "Setting SUPABASE_KEY (from SUPABASE_ANON_KEY)..."
                heroku config:set "SUPABASE_KEY=$value" -a "$APP_NAME"
            fi
            
            echo "Setting $key..."
            heroku config:set "$key=$value" -a "$APP_NAME"
        fi
    done < .env
    
    echo -e "${GREEN}✅ Environment variables configured${NC}"
else
    echo -e "${RED}⚠️  No .env file found. You'll need to set variables manually.${NC}"
    echo "Run: heroku config:set KEY=VALUE -a $APP_NAME"
fi

# Add git remote if not exists
echo -e "${BLUE}🔗 Configuring Git remote...${NC}"
if ! git remote | grep -q heroku; then
    heroku git:remote -a "$APP_NAME"
    echo -e "${GREEN}✅ Git remote added${NC}"
else
    echo -e "${GREEN}✅ Git remote already exists${NC}"
fi

# Deploy
echo -e "${BLUE}📤 Deploying to Heroku...${NC}"
echo "This may take a few minutes..."

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Commit any changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Committing changes..."
    git add .
    git commit -m "Deploy to Heroku" || echo "Nothing to commit"
fi

# Push to Heroku (use current branch)
if [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "master" ]; then
    git push heroku "$CURRENT_BRANCH"
else
    echo "Pushing $CURRENT_BRANCH to Heroku main..."
    git push heroku "$CURRENT_BRANCH:main"
fi

# Scale dyno
echo -e "${BLUE}⚡ Scaling web dyno...${NC}"
heroku ps:scale web=1 -a "$APP_NAME"

# Show app info
echo ""
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo "================================"

# Wait a moment for app to start
echo "Waiting for app to start..."
sleep 5

# Check dyno status
echo -e "${BLUE}📊 Dyno Status:${NC}"
heroku ps -a "$APP_NAME"

echo ""
heroku info -a "$APP_NAME"

# Get app URL
APP_URL=$(heroku info -a "$APP_NAME" -s | grep web_url | cut -d= -f2)

echo ""
echo -e "${GREEN}🎉 Your API is live at:${NC}"
echo -e "${BLUE}${APP_URL}${NC}"
echo ""

# Test health endpoint
echo -e "${BLUE}🏥 Testing health endpoint...${NC}"
if curl -sf "${APP_URL}health" > /dev/null; then
    echo -e "${GREEN}✅ API is responding!${NC}"
    curl -s "${APP_URL}health" | jq . 2>/dev/null || curl -s "${APP_URL}health"
else
    echo -e "${RED}⚠️  API not responding yet. Check logs:${NC}"
    echo "heroku logs --tail -a $APP_NAME"
fi

echo ""
echo -e "${GREEN}� Useful commands:${NC}"
echo -e "${BLUE}Test API:${NC} curl ${APP_URL}health"
echo -e "${BLUE}View logs:${NC} heroku logs --tail -a $APP_NAME"
echo -e "${BLUE}Restart:${NC} heroku restart -a $APP_NAME"
echo -e "${BLUE}Scale up:${NC} heroku ps:scale web=2 -a $APP_NAME"
echo -e "${BLUE}Config:${NC} heroku config -a $APP_NAME"
