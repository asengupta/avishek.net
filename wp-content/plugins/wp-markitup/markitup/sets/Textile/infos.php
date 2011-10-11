<?php
	if (!class_exists('TextileTagSet', false)) {
		class TextileTagSet extends TagSet {
		
			function getTagSetName() {
				return 'Textile';
			}
		
			function getExternalPluginInfos() {
				return array(
					'name' => 'Textile 2 (Improved)',
					'website' => 'http://wordpress.org/extend/plugins/textile-2/'
				);
			}
		
			function isExternalPluginActive() {
				return class_exists('Textile2_New');
			}
		
			function isExternalPluginInstalled() {
				return is_dir($this->getPluginDir().'textile-2');
			}
		}
	}
	$tagset = new TextileTagSet();
?>