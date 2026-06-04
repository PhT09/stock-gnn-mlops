FROM node:22-alpine

WORKDIR /app

# Cache layer: chỉ rebuild khi package.json đổi
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

EXPOSE 5173

# Vite dev với host 0.0.0.0 + polling (hot reload qua volume mount)
CMD ["npm", "run", "dev"]
