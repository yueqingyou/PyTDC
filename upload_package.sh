#!/bin/bash

# PyTDC Package Upload Script
# This script handles the complete workflow for uploading the PyTDC package to PyPI

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    local missing_deps=()
    
    if ! command_exists python3; then
        missing_deps+=("python3")
    fi
    
    if ! command_exists pip; then
        missing_deps+=("pip")
    fi
    
    # Check if twine is installed
    if ! python3 -c "import twine" 2>/dev/null; then
        print_warning "twine not found, installing..."
        pip install twine
    fi
    
    # Check if build is installed
    if ! python3 -c "import build" 2>/dev/null; then
        print_warning "build not found, installing..."
        pip install build
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi
    
    print_success "All dependencies are available"
}

# Function to clean previous builds
clean_build() {
    print_status "Cleaning previous builds..."
    
    # Remove build artifacts
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    
    # Remove Python cache
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    print_success "Build artifacts cleaned"
}

# Function to build the package
build_package() {
    print_status "Building package..."
    
    # Build source distribution and wheel
    python3 -m build
    
    if [ $? -eq 0 ]; then
        print_success "Package built successfully"
    else
        print_error "Package build failed"
        exit 1
    fi
}

# Function to check the built package
check_package() {
    print_status "Checking package with twine..."
    
    # Check the built package
    python3 -m twine check dist/*
    
    if [ $? -eq 0 ]; then
        print_success "Package check passed"
    else
        print_error "Package check failed"
        exit 1
    fi
}

# Function to show package contents
show_package_info() {
    print_status "Package information:"
    echo
    echo "Built files:"
    ls -la dist/
    echo
    
    if [ -f "dist/*.tar.gz" ]; then
        print_status "Source distribution contents:"
        tar -tzf dist/*.tar.gz | head -20
        if [ $(tar -tzf dist/*.tar.gz | wc -l) -gt 20 ]; then
            echo "... (showing first 20 files)"
        fi
        echo
    fi
}

# Function to configure credentials if needed
setup_credentials() {
    local repo="$1"
    
    # Check if credentials are already configured
    if [ -n "$TWINE_USERNAME" ] && [ -n "$TWINE_PASSWORD" ]; then
        return 0
    fi
    
    # Check if .pypirc exists and has the repository
    if [ -f ~/.pypirc ]; then
        if grep -q "\[$repo\]" ~/.pypirc; then
            return 0
        fi
    fi
    
    # Prompt for token configuration
    print_warning "No credentials configured for $repo"
    echo
    print_status "You can configure credentials in several ways:"
    echo
    echo "1. Environment variables (recommended for CI/CD):"
    echo "   export TWINE_USERNAME=__token__"
    echo "   export TWINE_PASSWORD=your-pypi-token-here"
    echo
    echo "2. .pypirc file (recommended for local development):"
    echo "   The script can help you create this file"
    echo
    echo "3. Interactive input (less secure):"
    echo "   Enter credentials when prompted"
    echo
    read -p "Would you like to set up .pypirc configuration? (y/n): " setup_config
    
    if [ "$setup_config" = "y" ] || [ "$setup_config" = "Y" ]; then
        setup_pypirc "$repo"
    else
        print_status "Please configure credentials manually and run the script again"
        return 1
    fi
}

# Function to set up .pypirc
setup_pypirc() {
    local repo="$1"
    
    print_status "Setting up .pypirc configuration for $repo"
    echo
    print_warning "Your token will be stored in ~/.pypirc (secured with 600 permissions)"
    echo
    
    read -p "Enter your PyPI API token (starts with pypi-): " -s token
    echo
    
    if [ -z "$token" ]; then
        print_error "Token cannot be empty"
        return 1
    fi
    
    # Create or update .pypirc
    local pypirc_path="$HOME/.pypirc"
    
    # Backup existing .pypirc if it exists
    if [ -f "$pypirc_path" ]; then
        cp "$pypirc_path" "$pypirc_path.backup.$(date +%Y%m%d_%H%M%S)"
        print_status "Backed up existing .pypirc"
    fi
    
    # Create new .pypirc
    cat > "$pypirc_path" << EOF
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = $token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = $token
EOF
    
    # Secure the file
    chmod 600 "$pypirc_path"
    
    print_success ".pypirc configured successfully"
}

# Function to upload to TestPyPI
upload_test() {
    print_status "Uploading to TestPyPI..."
    
    # Set up credentials if needed
    if ! setup_credentials "testpypi"; then
        return 1
    fi
    
    # Upload to TestPyPI
    python3 -m twine upload --repository testpypi dist/*
    
    if [ $? -eq 0 ]; then
        print_success "Successfully uploaded to TestPyPI"
        print_status "You can test install with:"
        echo "  pip install --index-url https://test.pypi.org/simple/ pytdc-nextml"
    else
        print_error "Upload to TestPyPI failed"
        exit 1
    fi
}

# Function to upload to PyPI
upload_prod() {
    print_status "Uploading to PyPI..."
    
    # Set up credentials if needed
    if ! setup_credentials "pypi"; then
        return 1
    fi
    
    # Final confirmation
    echo -e "${YELLOW}WARNING: This will upload to the production PyPI!${NC}"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        print_status "Upload cancelled"
        exit 0
    fi
    
    # Upload to PyPI
    python3 -m twine upload dist/*
    
    if [ $? -eq 0 ]; then
        print_success "Successfully uploaded to PyPI"
        print_status "Package is now available at:"
        echo "  https://pypi.org/project/pytdc-nextml/"
    else
        print_error "Upload to PyPI failed"
        exit 1
    fi
}

# Function to show usage
show_help() {
    echo "PyTDC Package Upload Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --test-only     Build and upload to TestPyPI only"
    echo "  --prod-only     Upload to PyPI (requires existing build)"
    echo "  --check-only    Build and check package only"
    echo "  --clean-only    Clean build artifacts only"
    echo "  --help          Show this help message"
    echo
    echo "Default behavior: Clean, build, check, and prompt for upload destination"
    echo
    echo "Environment Variables:"
    echo "  TWINE_USERNAME  PyPI/TestPyPI username"
    echo "  TWINE_PASSWORD  PyPI/TestPyPI password or API token"
    echo "  TWINE_REPOSITORY_URL  Custom repository URL"
    echo
    echo "Examples:"
    echo "  $0                    # Full workflow with prompts"
    echo "  $0 --test-only        # Build and upload to TestPyPI"
    echo "  $0 --check-only       # Build and check only"
    echo "  $0 --clean-only       # Clean build artifacts"
}

# Main workflow
main() {
    echo "========================================="
    echo "       PyTDC Package Upload Script      "
    echo "========================================="
    echo
    
    # Parse command line arguments
    case "${1:-}" in
        --help|-h)
            show_help
            exit 0
            ;;
        --clean-only)
            check_dependencies
            clean_build
            exit 0
            ;;
        --check-only)
            check_dependencies
            clean_build
            build_package
            check_package
            show_package_info
            exit 0
            ;;
        --test-only)
            check_dependencies
            clean_build
            build_package
            check_package
            show_package_info
            upload_test
            exit 0
            ;;
        --prod-only)
            if [ ! -d "dist" ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
                print_error "No built packages found in dist/. Run build first."
                exit 1
            fi
            check_dependencies
            check_package
            upload_prod
            exit 0
            ;;
        "")
            # Default workflow
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    
    # Default full workflow
    check_dependencies
    clean_build
    build_package
    check_package
    show_package_info
    
    # Prompt for upload destination
    echo
    print_status "Package is ready for upload. Choose destination:"
    echo "1) TestPyPI (recommended for testing)"
    echo "2) PyPI (production)"
    echo "3) Skip upload"
    echo
    read -p "Enter your choice (1-3): " choice
    
    case $choice in
        1)
            upload_test
            ;;
        2)
            upload_prod
            ;;
        3)
            print_status "Upload skipped. Built packages are available in dist/"
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
